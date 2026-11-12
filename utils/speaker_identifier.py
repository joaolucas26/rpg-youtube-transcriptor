import os
import re
import torch
import numpy as np
import torchaudio
from pydub import AudioSegment


if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


def load_speaker_model():
    print("Loading Speaker Recognition model (SpeechBrain)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: GPU not found. Process will be very slow.")

    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY  # Evita erro de permissão de atalho (symlink) no Windows
    )
    return speaker_model


def _encode_chunk_tensor(speech_chunk: AudioSegment, speaker_model) -> torch.Tensor:
    audio_16k = speech_chunk.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    samples = np.array(audio_16k.get_array_of_samples())
    audio_tensor = torch.from_numpy(samples).float() / 32768.0
    if len(audio_tensor) < 3200:
        repeats = (3200 // max(len(audio_tensor), 1)) + 1
        audio_tensor = audio_tensor.repeat(repeats)
    emb = speaker_model.encode_batch(audio_tensor.unsqueeze(0)).squeeze()
    return torch.nn.functional.normalize(emb, dim=0)


def _encode_audio_windows(audio_16k: AudioSegment, speaker_model, window_ms=4000, step_ms=2000) -> list[torch.Tensor]:
    """
    Fatia áudios longos (ex: áudios de WhatsApp de 20~30s) em janelas de 4s,
    ignorando janelas de silêncio e extraindo embeddings limpos de cada trecho.
    """
    total_ms = len(audio_16k)
    if total_ms <= window_ms + 500:
        return [_encode_chunk_tensor(audio_16k, speaker_model)]

    window_embeddings = []
    for start in range(0, max(1, total_ms - window_ms + 1), step_ms):
        chunk = audio_16k[start : start + window_ms]
        if chunk.rms < 150:
            continue
        window_embeddings.append(_encode_chunk_tensor(chunk, speaker_model))

    if not window_embeddings:
        return [_encode_chunk_tensor(audio_16k, speaker_model)]

    return window_embeddings


def load_character_profiles(
    speaker_model,
    samples_folder="inputs/sample_audios",
    use_samples=True,
    debias_channel_weight=0.30,
):
    """
    Carrega os perfis de voz da pasta de samples (ex: áudios de WhatsApp, celular, etc.).
    Se `use_samples=False` ou a pasta estiver vazia, retorna dicionário vazio para
    descoberta 100% automática no Warm-up da sessão.
    """
    if not use_samples:
        print("ℹ️ Uso de samples desativado (`USE_VOICE_SAMPLES=False`). Modo de descoberta automática ativo.")
        return {}

    print(f"Loading character voice profiles from '{samples_folder}'...")
    grouped_embeddings: dict[str, list[torch.Tensor]] = {}
    grouped_file_means: dict[str, list[torch.Tensor]] = {}

    if os.path.exists(samples_folder):
        with torch.no_grad():
            for filename in sorted(os.listdir(samples_folder)):
                if not filename.endswith((".ogg", ".wav", ".flac", ".mp3")):
                    continue

                raw_name = os.path.splitext(filename)[0]
                character_name = re.sub(r"_\d+$", "", raw_name)
                filepath = os.path.join(samples_folder, filename)

                audio = AudioSegment.from_file(filepath)
                audio_16k = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

                win_embs = _encode_audio_windows(audio_16k, speaker_model)
                file_mean = torch.nn.functional.normalize(
                    torch.stack(win_embs, dim=0).mean(dim=0), dim=0
                )
                grouped_embeddings.setdefault(character_name, []).extend(win_embs)
                grouped_file_means.setdefault(character_name, []).append(file_mean)
                print(
                    f"Amostra carregada: {filename} -> perfil '{character_name}' "
                    f"({len(win_embs)} janelas de voz)"
                )

    raw_anchors = {}
    for character_name, file_means in grouped_file_means.items():
        mean_emb = torch.stack(file_means, dim=0).mean(dim=0)
        raw_anchors[character_name] = torch.nn.functional.normalize(mean_emb, dim=0)

    # Subtrai parcialmente a média comum do canal (ex: assinatura acústica do WhatsApp)
    channel_bias = None
    if len(raw_anchors) >= 3 and debias_channel_weight > 0:
        channel_bias = torch.stack(list(raw_anchors.values()), dim=0).mean(dim=0)

    character_profiles = {}
    for character_name, embeddings in grouped_embeddings.items():
        anchor_emb = raw_anchors[character_name]
        if channel_bias is not None:
            anchor_emb = anchor_emb - (debias_channel_weight * channel_bias)
            anchor_emb = torch.nn.functional.normalize(anchor_emb, dim=0)

        debiased_samples = []
        for emb in embeddings:
            s_emb = emb - (debias_channel_weight * channel_bias) if channel_bias is not None else emb
            debiased_samples.append(torch.nn.functional.normalize(s_emb, dim=0))

        character_profiles[character_name] = {
            "anchor": anchor_emb,
            "current": anchor_emb.clone(),
            "samples": debiased_samples,
            "session_samples": [],
        }
        print(f"Perfil consolidado: {character_name} ({len(embeddings)} janela(s))")

    if not character_profiles:
        print("ℹ️ Nenhum sample prévio encontrado. Os speakers serão descobertos automaticamente no Warm-up.")

    return character_profiles


def warmup_session_profiles(
    audio: AudioSegment,
    speech_timestamps: list[dict],
    speaker_model,
    character_profiles: dict,
    max_speakers: int = 6,
    max_warmup_segments: int = 180,
    min_segment_ms: int = 3000,
):
    """
    Warm-up Inteligente (Pré-Análise de Voz da Sessão):
    Analisa uma amostra de falas longas (>= 3s) distribuídas pelo vídeo inteiro ANTES de rodar o Whisper:
    1. Se o cliente enviou samples (de qualquer origem: WhatsApp, celular, etc.), usa normalização Z-Score
       para encontrar as falas reais de cada personagem no vídeo, calibrando os perfis nativos da sessão
       antes mesmo da 1ª frase ser transcrita.
    2. Se ainda houver vagas até `max_speakers` (ou se o cliente NÃO enviou nenhum sample), agrupa
       automaticamente as vozes restantes da sessão em clusters (`speaker_1..N`).
    """
    print("--- Executando Warm-up Inteligente de Vozes na Sessão ---")
    long_segments = []
    for ts in speech_timestamps:
        start_ms = ts["start"] / 16.0
        end_ms = ts["end"] / 16.0
        if (end_ms - start_ms) >= min_segment_ms:
            long_segments.append((start_ms, end_ms))

    if not long_segments:
        print("⚠️ Nenhuma fala longa (>= 3s) encontrada para o Warm-up.")
        return character_profiles

    if len(long_segments) > max_warmup_segments:
        indices = np.linspace(0, len(long_segments) - 1, max_warmup_segments, dtype=int)
        sampled_segments = [long_segments[i] for i in indices]
    else:
        sampled_segments = long_segments

    print(f"Extraindo impressões de voz de {len(sampled_segments)} trechos longos do vídeo...")
    session_embs = []
    with torch.no_grad():
        for start_ms, end_ms in sampled_segments:
            chunk = audio[start_ms:end_ms]
            session_embs.append(_encode_chunk_tensor(chunk, speaker_model))

    E = torch.stack(session_embs, dim=0)  # (N, D)
    N = E.shape[0]
    assigned_mask = torch.zeros(N, dtype=torch.bool, device=E.device)

    # 1. Alinhamento de Domínio para perfis enviados pelo cliente (WhatsApp/Celular -> Sessão)
    known_names = list(character_profiles.keys())
    if known_names:
        K = len(known_names)
        raw_scores = torch.zeros((K, N), device=E.device)
        for k, name in enumerate(known_names):
            prof = character_profiles[name]
            anchor_sim = torch.mv(E, prof["anchor"])
            win_sims = [torch.mv(E, s) for s in prof.get("samples", [])]
            best_win_sim = torch.stack(win_sims, dim=0).max(dim=0).values if win_sims else anchor_sim
            raw_scores[k] = 0.5 * anchor_sim + 0.5 * best_win_sim

        # Normalização Z-Score por personagem: elimina a desvantagem de microfones diferentes
        # (ex: quando 'tyr' fala na sessão, o score dele sobe muito acima da própria média dele)
        mean_k = raw_scores.mean(dim=1, keepdim=True)
        std_k = raw_scores.std(dim=1, keepdim=True).clamp(min=1e-4)
        z_scores = (raw_scores - mean_k) / std_k

        best_raw_vals, best_raw_idx = raw_scores.max(dim=0)
        sorted_raw = torch.sort(raw_scores, dim=0, descending=True).values
        raw_margin = sorted_raw[0] - sorted_raw[1] if K > 1 else sorted_raw[0]

        best_z_vals, best_k_idx = z_scores.max(dim=0)
        sorted_z = torch.sort(z_scores, dim=0, descending=True).values
        z_margin = sorted_z[0] - sorted_z[1] if K > 1 else sorted_z[0]

        for k, name in enumerate(known_names):
            # Seleciona apenas os trechos onde o personagem venceu TANTO no Score Bruto
            # quanto no Z-Score (evita que quem falou pouco pegue falas de quem falou muito)
            candidate_mask = (
                (best_raw_idx == k)
                & (best_k_idx == k)
                & (raw_scores[k] >= 0.25)
                & (raw_margin >= 0.04)
                & (best_z_vals >= 0.8)
                & (z_margin >= 0.35)
            )
            cand_indices = torch.where(candidate_mask)[0]

            if len(cand_indices) < 2:
                fallback_mask = (
                    (best_raw_idx == k)
                    & (best_k_idx == k)
                    & (raw_scores[k] >= 0.22)
                    & (raw_margin >= 0.02)
                    & (best_z_vals >= 0.5)
                )
                cand_indices = torch.where(fallback_mask)[0]

            if len(cand_indices) > 0:
                # Ordena pelos maiores Z-scores daquele personagem e pega até 8 falas nativas da sessão
                top_order = torch.argsort(z_scores[k, cand_indices], descending=True)[:8]
                chosen_idx = cand_indices[top_order]
                assigned_mask[chosen_idx] = True

                native_embs = [E[idx].clone() for idx in chosen_idx]
                native_mean = torch.nn.functional.normalize(
                    torch.stack(native_embs, dim=0).mean(dim=0), dim=0
                )

                prof = character_profiles[name]
                prof["session_samples"] = native_embs
                # Perfil ativo passa a ser 85% voz nativa da sessão + 15% sample original enviado
                prof["current"] = torch.nn.functional.normalize(
                    0.85 * native_mean + 0.15 * prof["anchor"], dim=0
                )
                print(
                    f"  ✅ '{name}' alinhado ao áudio da sessão com {len(native_embs)} amostra(s) nativas "
                    f"(score inicial médio: {raw_scores[k, chosen_idx].mean().item():.2f})"
                )
            else:
                print(f"  ℹ️ '{name}': nenhuma fala longa clara encontrada no Warm-up; mantendo sample original.")

    # 2. Descoberta Automática de Clusters para vagas abertas (modo sem samples ou samples parciais)
    open_slots = max_speakers - len(character_profiles)
    if open_slots > 0:
        unassigned_idx = torch.where(~assigned_mask)[0]
        if len(known_names) > 0 and len(unassigned_idx) > 0:
            # Filtra apenas trechos que realmente não parecem com nenhum perfil já conhecido (< 0.28)
            known_currents = torch.stack([character_profiles[n]["current"] for n in known_names], dim=0)
            max_known_sim = torch.mm(E[unassigned_idx], known_currents.t()).max(dim=1).values
            unassigned_idx = unassigned_idx[max_known_sim < 0.28]

        if len(unassigned_idx) >= 3:
            U = E[unassigned_idx]
            sim_matrix = torch.mm(U, U.t())
            used = torch.zeros(len(unassigned_idx), dtype=torch.bool, device=E.device)

            for _ in range(open_slots):
                avail = torch.where(~used)[0]
                if len(avail) < 3:
                    break

                # Escolhe o centro com maior densidade de vizinhos próximos (similaridade >= 0.42)
                sub_sim = sim_matrix[avail][:, avail]
                neighbors_count = (sub_sim >= 0.42).sum(dim=1)
                best_local = torch.argmax(neighbors_count).item()
                if neighbors_count[best_local].item() < 3:
                    break

                seed_idx = avail[best_local]
                cluster_members = avail[sim_matrix[seed_idx, avail] >= 0.40]
                used[cluster_members] = True

                # Refina o centroide do cluster descoberto
                cluster_embs = U[cluster_members]
                centroid = torch.nn.functional.normalize(cluster_embs.mean(dim=0), dim=0)
                top_cluster_order = torch.argsort(
                    torch.mv(cluster_embs, centroid), descending=True
                )[:8]
                best_cluster_embs = [cluster_embs[i].clone() for i in top_cluster_order]
                centroid = torch.nn.functional.normalize(
                    torch.stack(best_cluster_embs, dim=0).mean(dim=0), dim=0
                )

                unknown_count = sum(1 for name in character_profiles if name.startswith("speaker_"))
                new_speaker_name = f"speaker_{unknown_count + 1}"
                character_profiles[new_speaker_name] = {
                    "anchor": centroid,
                    "current": centroid.clone(),
                    "samples": best_cluster_embs,
                    "session_samples": best_cluster_embs,
                }
                print(
                    f"  🆕 Cluster descoberto no Warm-up ({len(character_profiles)}/{max_speakers}): "
                    f"'{new_speaker_name}' ({len(cluster_members)} falas agrupadas)"
                )

    print("--- Warm-up de Vozes concluído! ---\n")
    return character_profiles


def identify_speaker(
    speech_chunk,
    speaker_model,
    character_profiles,
    max_speakers=6,
    min_score_threshold=0.15,
    new_speaker_threshold=0.24,
    min_ms_for_new_speaker=2500,
    max_session_adaptations=10,
    analytics=None,
    chunk_index=0,
    start_ms=0.0,
    end_ms=0.0,
):
    """
    Identifica quem está falando comparando com os perfis já alinhados no Warm-up.
    """
    with torch.no_grad():
        chunk_embedding = _encode_chunk_tensor(speech_chunk, speaker_model)

        best_speaker = "ignored"
        best_score = -1.0
        second_best_speaker = "none"
        second_best_score = -1.0
        all_scores: dict[str, dict[str, float]] = {}

        for character_name, profile_data in character_profiles.items():
            if isinstance(profile_data, dict):
                anchor_score = torch.dot(chunk_embedding, profile_data["anchor"]).item()
                current_score = torch.dot(chunk_embedding, profile_data["current"]).item()
                session_embs = profile_data.get("session_samples", [])

                if session_embs:
                    best_session = max(torch.dot(chunk_embedding, s).item() for s in session_embs)
                    score = 0.65 * max(current_score, best_session) + 0.35 * anchor_score
                else:
                    best_wa_win = max(
                        (torch.dot(chunk_embedding, s).item() for s in profile_data.get("samples", [])),
                        default=anchor_score,
                    )
                    score = 0.50 * anchor_score + 0.50 * best_wa_win

                all_scores[character_name] = {
                    "anchor": anchor_score,
                    "current": current_score,
                    "best": score,
                }
            else:
                score = torch.dot(chunk_embedding, profile_data).item()
                all_scores[character_name] = {
                    "anchor": score,
                    "current": score,
                    "best": score,
                }

            if score > best_score:
                second_best_score = best_score
                second_best_speaker = best_speaker
                best_score = score
                best_speaker = character_name
            elif score > second_best_score:
                second_best_score = score
                second_best_speaker = character_name

        chosen_speaker = best_speaker
        was_adapted = False

        has_open_slots = len(character_profiles) < max_speakers
        required_ms = 1500 if len(character_profiles) == 0 else min_ms_for_new_speaker

        if has_open_slots and best_score < new_speaker_threshold:
            if len(speech_chunk) >= required_ms:
                unknown_count = sum(1 for name in character_profiles if name.startswith("speaker_"))
                new_speaker_name = f"speaker_{unknown_count + 1}"
                character_profiles[new_speaker_name] = {
                    "anchor": chunk_embedding,
                    "current": chunk_embedding.clone(),
                    "samples": [chunk_embedding.clone()],
                    "session_samples": [chunk_embedding.clone()],
                }
                all_scores[new_speaker_name] = {"anchor": 1.0, "current": 1.0, "best": 1.0}
                print(
                    f"🆕 Novo perfil salvo em memória ({len(character_profiles)}/{max_speakers}): "
                    f"{new_speaker_name} (melhor score anterior: {best_score:.2f})"
                )
                chosen_speaker = new_speaker_name
            else:
                chosen_speaker = "ignored"
        elif best_score < min_score_threshold:
            chosen_speaker = "ignored"
        else:
            profile = character_profiles.get(best_speaker)
            if isinstance(profile, dict):
                session_embs = profile.setdefault("session_samples", [])
                n_calib = len(session_embs)
                margin = best_score - second_best_score

                if n_calib < 3:
                    can_adapt = (
                        len(speech_chunk) >= 2200
                        and best_score >= 0.27
                        and margin >= 0.06
                    )
                elif n_calib < max_session_adaptations:
                    can_adapt = (
                        len(speech_chunk) >= 3000
                        and best_score >= 0.45
                        and margin >= 0.14
                    )
                else:
                    can_adapt = False

                if can_adapt:
                    session_embs.append(chunk_embedding.clone())
                    session_mean = torch.stack(session_embs, dim=0).mean(dim=0)
                    session_mean = torch.nn.functional.normalize(session_mean, dim=0)
                    adapted = 0.85 * session_mean + 0.15 * profile["anchor"]
                    profile["current"] = torch.nn.functional.normalize(adapted, dim=0)
                    was_adapted = True

        if analytics is not None:
            analytics.log_chunk(
                chunk_index=chunk_index,
                start_ms=start_ms,
                end_ms=end_ms,
                chosen_speaker=chosen_speaker,
                best_speaker_candidate=best_speaker,
                best_score=best_score,
                second_best_speaker=second_best_speaker,
                second_best_score=second_best_score,
                all_scores=all_scores,
                was_adapted=was_adapted,
                speech_chunk=speech_chunk,
            )

        return chosen_speaker
