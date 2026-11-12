import os
import csv
import json
from collections import defaultdict
import torch
from pydub import AudioSegment
from utils.utils import format_timestamp


class SessionAnalytics:
    def __init__(self, samples_folder: str = "inputs/sample_audios"):
        self.samples_folder = samples_folder
        self.sample_durations: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self.initial_cross_similarity: dict[str, dict[str, float]] = {}
        self.records: list[dict] = []
        self.adaptation_counts: dict[str, int] = defaultdict(int)
        # Guarda o melhor AudioSegment (>= 3.5s) de cada speaker para exportar como sample sugerido
        self.best_audio_clips: dict[str, tuple[float, float, AudioSegment, str]] = {}

    def inspect_samples(self, character_profiles: dict):
        """
        Analisa a duração dos arquivos em sample_audios e calcula a matriz de
        similaridade inicial entre os perfis para detectar vozes parecidas ou samples curtos.
        """
        if os.path.exists(self.samples_folder):
            for filename in sorted(os.listdir(self.samples_folder)):
                if not filename.endswith((".ogg", ".wav", ".flac", ".mp3")):
                    continue
                filepath = os.path.join(self.samples_folder, filename)
                try:
                    audio = AudioSegment.from_file(filepath)
                    duration_sec = len(audio) / 1000.0
                    raw_name = os.path.splitext(filename)[0]
                    char_name = (
                        raw_name.rsplit("_", 1)[0]
                        if "_" in raw_name and raw_name.rsplit("_", 1)[1].isdigit()
                        else raw_name
                    )
                    self.sample_durations[char_name].append((filename, duration_sec))
                except Exception:
                    pass

        names = list(character_profiles.keys())
        for n1 in names:
            self.initial_cross_similarity[n1] = {}
            emb1 = (
                character_profiles[n1]["anchor"]
                if isinstance(character_profiles[n1], dict)
                else character_profiles[n1]
            )
            for n2 in names:
                emb2 = (
                    character_profiles[n2]["anchor"]
                    if isinstance(character_profiles[n2], dict)
                    else character_profiles[n2]
                )
                sim = torch.dot(emb1, emb2).item()
                self.initial_cross_similarity[n1][n2] = round(sim, 4)

    def log_chunk(
        self,
        chunk_index: int,
        start_ms: float,
        end_ms: float,
        chosen_speaker: str,
        best_speaker_candidate: str,
        best_score: float,
        second_best_speaker: str,
        second_best_score: float,
        all_scores: dict[str, dict[str, float]],
        was_adapted: bool,
        speech_chunk: AudioSegment | None = None,
    ):
        duration_ms = end_ms - start_ms
        duration_sec = duration_ms / 1000.0
        margin = best_score - second_best_score
        ts_str = format_timestamp(start_ms / 1000.0)

        if was_adapted:
            self.adaptation_counts[chosen_speaker] += 1

        # Guarda o melhor corte de áudio (>= 3.5s) de cada speaker para exportar em suggested_samples/
        if (
            speech_chunk is not None
            and chosen_speaker != "ignored"
            and duration_sec >= 3.5
        ):
            quality_metric = best_score + (0.5 * max(margin, 0.0))
            prev = self.best_audio_clips.get(chosen_speaker)
            if prev is None or quality_metric > prev[0]:
                self.best_audio_clips[chosen_speaker] = (
                    quality_metric,
                    best_score,
                    speech_chunk,
                    ts_str,
                )

        self.records.append(
            {
                "chunk_index": chunk_index,
                "start_sec": round(start_ms / 1000.0, 2),
                "timestamp": ts_str,
                "duration_sec": round(duration_sec, 2),
                "chosen_speaker": chosen_speaker,
                "best_candidate": best_speaker_candidate,
                "best_score": round(best_score, 4),
                "second_best_speaker": second_best_speaker,
                "second_best_score": round(second_best_score, 4),
                "margin": round(margin, 4),
                "was_adapted": was_adapted,
                "all_scores": all_scores,
            }
        )

    def save_reports(self, output_folder: str, character_profiles: dict):
        """
        Gera na pasta de saída do vídeo:
        1. analytics_report.txt (Relatório legível com diagnósticos e recomendações)
        2. analytics_chunks.csv (Tabela detalhada trecho a trecho com todos os scores)
        3. analytics_summary.json (Dados estruturados)
        4. suggested_samples/<speaker>.ogg (Melhores trechos de voz recortados automaticamente)
        """
        if not self.records:
            print("⚠️ Nenhum dado de analytics registrado para salvar.")
            return

        os.makedirs(output_folder, exist_ok=True)
        report_txt_path = os.path.join(output_folder, "analytics_report.txt")
        csv_path = os.path.join(output_folder, "analytics_chunks.csv")
        json_path = os.path.join(output_folder, "analytics_summary.json")
        suggested_folder = os.path.join(output_folder, "suggested_samples")

        # Exporta automaticamente o melhor sample de áudio de cada speaker
        if self.best_audio_clips:
            os.makedirs(suggested_folder, exist_ok=True)
            for spk_name, (_, sc, clip, _) in self.best_audio_clips.items():
                clip_path = os.path.join(suggested_folder, f"{spk_name}.ogg")
                try:
                    clip.export(clip_path, format="ogg")
                except Exception:
                    pass

        known_characters = list(character_profiles.keys())
        total_chunks = len(self.records)
        total_duration_sec = sum(r["duration_sec"] for r in self.records)

        # Estatísticas por speaker escolhido
        speaker_stats = {}
        for name in known_characters + ["ignored"]:
            chosen_recs = [r for r in self.records if r["chosen_speaker"] == name]
            count = len(chosen_recs)
            dur = sum(r["duration_sec"] for r in chosen_recs)
            avg_win_score = (
                sum(r["best_score"] for r in chosen_recs) / count if count > 0 else 0.0
            )
            avg_margin = (
                sum(r["margin"] for r in chosen_recs) / count if count > 0 else 0.0
            )

            anchor_scores_all = [
                r["all_scores"][name]["anchor"]
                for r in self.records
                if name in r["all_scores"]
            ]
            anchor_scores_when_won = [
                r["all_scores"][name]["anchor"]
                for r in chosen_recs
                if name in r["all_scores"]
            ]
            current_scores_when_won = [
                r["all_scores"][name]["current"]
                for r in chosen_recs
                if name in r["all_scores"]
            ]

            drift_similarity = 1.0
            if name in character_profiles and isinstance(character_profiles[name], dict):
                drift_similarity = torch.dot(
                    character_profiles[name]["anchor"],
                    character_profiles[name]["current"],
                ).item()

            speaker_stats[name] = {
                "chunks_count": count,
                "chunks_pct": round((count / total_chunks) * 100, 1) if total_chunks else 0.0,
                "duration_sec": round(dur, 1),
                "duration_pct": round((dur / total_duration_sec) * 100, 1) if total_duration_sec else 0.0,
                "avg_win_score": round(avg_win_score, 3),
                "avg_margin": round(avg_margin, 3),
                "avg_anchor_global": round(sum(anchor_scores_all) / len(anchor_scores_all), 3) if anchor_scores_all else 0.0,
                "avg_anchor_when_won": round(sum(anchor_scores_when_won) / len(anchor_scores_when_won), 3) if anchor_scores_when_won else 0.0,
                "avg_current_when_won": round(sum(current_scores_when_won) / len(current_scores_when_won), 3) if current_scores_when_won else 0.0,
                "adaptations": self.adaptation_counts.get(name, 0),
                "anchor_to_current_sim": round(drift_similarity, 3),
            }

        # Pares de confusão (quando margin < 0.04 em trechos válidos)
        confusion_pairs = defaultdict(int)
        low_margin_examples = []
        for r in self.records:
            if r["chosen_speaker"] != "ignored" and r["margin"] < 0.04:
                pair = tuple(sorted([r["best_candidate"], r["second_best_speaker"]]))
                confusion_pairs[pair] += 1
                if r["duration_sec"] >= 2.5:
                    low_margin_examples.append(r)

        low_margin_examples.sort(key=lambda x: x["margin"])

        best_candidates_for_samples = {}
        for name in known_characters:
            candidates = [
                r for r in self.records
                if r["chosen_speaker"] == name and r["duration_sec"] >= 3.5
            ]
            candidates.sort(key=lambda x: (x["best_score"], x["margin"]), reverse=True)
            best_candidates_for_samples[name] = candidates[:3]

        # Salvar CSV trecho a trecho
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "chunk_index",
                "timestamp",
                "duration_sec",
                "chosen_speaker",
                "best_candidate",
                "best_score",
                "second_best_speaker",
                "second_best_score",
                "margin",
                "was_adapted",
            ]
            for name in known_characters:
                fieldnames.append(f"score_{name}_anchor")
                fieldnames.append(f"score_{name}_current")

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                row = {k: r[k] for k in fieldnames if k in r}
                for name in known_characters:
                    sc = r["all_scores"].get(name, {})
                    row[f"score_{name}_anchor"] = round(sc.get("anchor", 0.0), 4)
                    row[f"score_{name}_current"] = round(sc.get("current", 0.0), 4)
                writer.writerow(row)

        # Montar relatório em texto com diagnósticos automáticos
        lines = []
        lines.append("=" * 80)
        lines.append("📊 RELATÓRIO DE ANALYTICS — IDENTIFICAÇÃO DE SPEAKERS & QUALIDADE DE SAMPLES")
        lines.append("=" * 80)
        lines.append(f"Total de segmentos de fala analisados : {total_chunks}")
        lines.append(f"Tempo total de fala analisada         : {format_timestamp(total_duration_sec)} ({total_duration_sec:.1f}s)")
        lines.append("")

        # 1. Diagnóstico dos arquivos em sample_audios
        lines.append("-" * 80)
        lines.append("1. QUALIDADE INICIAL DOS SAMPLES (inputs/sample_audios)")
        lines.append("-" * 80)
        for name in known_characters:
            files_info = self.sample_durations.get(name, [])
            total_sample_sec = sum(d for _, d in files_info)
            files_str = ", ".join(f"{fname} ({d:.1f}s)" for fname, d in files_info) or "Descoberto na sessão (sem sample prévio)"
            status = "✅ OK"
            if total_sample_sec == 0:
                status = "🆕 DINÂMICO"
            elif total_sample_sec < 4.0:
                status = "⚠️ CURTO (<4s: ideal ter 6s a 15s de fala limpa)"
            elif total_sample_sec > 45.0:
                status = "ℹ️ LONGO (considere dividir em _1, _2 com trechos sem silêncio)"
            lines.append(f"  • {name:<12} | Total: {total_sample_sec:>5.1f}s | {status} | [{files_str}]")

        if self.initial_cross_similarity:
            lines.append("\n  Matriz de Similaridade Inicial entre os Samples (risco de confusão se > 0.35):")
            init_names = list(self.initial_cross_similarity.keys())
            header = "               " + "".join(f"{n:>10}" for n in init_names)
            lines.append(header)
            for n1 in init_names:
                row_str = f"  {n1:<12} "
                for n2 in init_names:
                    val = self.initial_cross_similarity.get(n1, {}).get(n2, 0.0)
                    row_str += f"{val:>10.2f}"
                lines.append(row_str)
        lines.append("")

        # 2. Participação e Desempenho por Personagem
        lines.append("-" * 80)
        lines.append("2. DESEMPENHO POR PERSONAGEM NA SESSÃO")
        lines.append("-" * 80)
        lines.append(
            f"  {'Speaker':<12} | {'Falas':>6} | {'Tempo (s)':>9} | {'% Tempo':>7} | "
            f"{'Score Méd':>9} | {'Sample Orig':>11} | {'Adaptado':>8} | {'Calibrações':>11}"
        )
        lines.append("  " + "-" * 76)
        for name in sorted(
            known_characters + ["ignored"],
            key=lambda x: speaker_stats[x]["duration_sec"],
            reverse=True,
        ):
            st = speaker_stats[name]
            lines.append(
                f"  {name:<12} | {st['chunks_count']:>6} | {st['duration_sec']:>8.1f}s | "
                f"{st['duration_pct']:>6.1f}% | {st['avg_win_score']:>9.3f} | "
                f"{st['avg_anchor_when_won']:>11.3f} | {st['avg_current_when_won']:>8.3f} | "
                f"{st['adaptations']:>11}"
            )
        lines.append("")

        # 3. Diagnósticos Automáticos (Gaps detectados)
        lines.append("-" * 80)
        lines.append("3. DIAGNÓSTICOS AUTOMÁTICOS & POSSÍVEIS GAPS")
        lines.append("-" * 80)
        diagnostics = []

        if "mestre" in speaker_stats:
            mestre_pct = speaker_stats["mestre"]["duration_pct"]
            max_char = max(known_characters, key=lambda c: speaker_stats[c]["duration_pct"])
            if max_char != "mestre" and mestre_pct < 20.0:
                diagnostics.append(
                    f"⚠️ ALERTA MESTRE POUCO UTILIZADO: 'mestre' teve apenas {mestre_pct:.1f}% do tempo de fala "
                    f"(quem mais falou foi '{max_char}' com {speaker_stats[max_char]['duration_pct']:.1f}%). "
                    f"Score médio do sample original do mestre: {speaker_stats['mestre']['avg_anchor_when_won']:.3f}."
                )

        for name in known_characters:
            st = speaker_stats[name]
            if st["chunks_count"] == 0 or st["duration_pct"] < 3.0:
                diagnostics.append(
                    f"⚠️ BAIXA DETECÇÃO ('{name}'): Detectado em apenas {st['chunks_count']} trechos ({st['duration_pct']:.1f}% do tempo) "
                    f"e {st['adaptations']} auto-calibrações. Ou a pessoa falou muito pouco na sessão, ou seu sample original "
                    f"(média global: {st['avg_anchor_global']:.3f}) está perdendo para outro jogador."
                )
            elif st["adaptations"] == 0 and st["chunks_count"] >= 5:
                diagnostics.append(
                    f"⚠️ SEM AUTO-CALIBRAÇÃO ('{name}'): Falou {st['chunks_count']} vezes, mas nenhuma fala atingiu os critérios "
                    f"de adaptação (score >= 0.45, margem >= 0.15 e duração >= 3.0s). O modelo nunca 'aprendeu' a voz dele na sessão."
                )
            elif st["avg_current_when_won"] - st["avg_anchor_when_won"] >= 0.08:
                diagnostics.append(
                    f"💡 SAMPLE ORIGINAL DEFASADO ('{name}'): O perfil auto-calibrado teve score médio {st['avg_current_when_won']:.3f}, "
                    f"bem acima do sample original ({st['avg_anchor_when_won']:.3f}). Substituir o sample de '{name}' pelo arquivo "
                    f"em 'suggested_samples/{name}.ogg' vai melhorar os primeiros minutos!"
                )

        if speaker_stats.get("ignored", {}).get("chunks_pct", 0.0) > 25.0:
            diagnostics.append(
                f"⚠️ MUITOS TRECHOS IGNORADOS ({speaker_stats['ignored']['chunks_pct']:.1f}% dos segmentos): "
                f"Muitas falas tiveram score < 0.15. Considere reduzir `min_score_threshold` para 0.12 ou melhorar os samples."
            )

        if not diagnostics:
            diagnostics.append("✅ Nenhuma anomalia grave detectada na distribuição de vozes.")

        for d in diagnostics:
            lines.append(f"  {d}")
        lines.append("")

        # 4. Pares de maior confusão
        lines.append("-" * 80)
        lines.append("4. PARES DE VOZES COM MAIOR DISPUTA (Margem < 0.04)")
        lines.append("-" * 80)
        if confusion_pairs:
            for (p1, p2), cnt in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  • {p1} vs {p2}: {cnt} falas com decisão apertada (margem < 0.04)")
        else:
            lines.append("  • Nenhuma disputa apertada registrada.")

        if low_margin_examples:
            lines.append("\n  Exemplos de falas longas (>=2.5s) com maior dúvida (para você ouvir no vídeo):")
            for ex in low_margin_examples[:8]:
                lines.append(
                    f"    [{ex['timestamp']}] ({ex['duration_sec']}s) Escolhido: {ex['best_candidate']} ({ex['best_score']:.3f}) "
                    f"vs 2º: {ex['second_best_speaker']} ({ex['second_best_score']:.3f}) | Margem: {ex['margin']:.3f}"
                )
        lines.append("")

        # 5. Sugestão de Timestamps para recortar novos samples
        lines.append("-" * 80)
        lines.append("5. MELHORES TIMESTAMPS DO VÍDEO (Exportados automaticamente em 'suggested_samples/')")
        lines.append("-" * 80)
        for name in known_characters:
            cands = best_candidates_for_samples.get(name, [])
            if cands:
                ts_list = ", ".join(
                    f"[{c['timestamp']} ({c['duration_sec']}s | score {c['best_score']:.2f})]"
                    for c in cands
                )
                lines.append(f"  • {name:<12}: {ts_list}")
            else:
                lines.append(f"  • {name:<12}: Nenhuma fala >= 3.5s com alta confiança encontrada.")
        lines.append("=" * 80)

        report_text = "\n".join(lines)
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_chunks": total_chunks,
                    "total_duration_sec": round(total_duration_sec, 1),
                    "initial_cross_similarity": self.initial_cross_similarity,
                    "speaker_stats": speaker_stats,
                    "diagnostics": diagnostics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\n📊 Relatório de Analytics salvo em: '{report_txt_path}'")
        print(f"📈 CSV detalhado de scores salvo em: '{csv_path}'")
        if self.best_audio_clips:
            print(f"🎙️ Melhores samples da sessão exportados em: '{suggested_folder}'")
