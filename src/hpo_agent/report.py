"""Report generation for HPO runs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from hpo_agent.models import ParamSpace, SearchSpaceChangeRecord, TrialRecord


def _format_param_space(param_space: ParamSpace) -> str:
    """ParamSpace を Markdown 箇条書き文字列にフォーマットする。"""
    lines = []
    for spec in param_space.specs:
        if spec.type == "categorical":
            lines.append(
                f"- **{spec.name}**: categorical, choices={list(spec.choices or [])}"
            )
        else:
            scale = "log スケール" if spec.log else "linear スケール"
            lines.append(
                f"- **{spec.name}**: {spec.type}, [{spec.low}, {spec.high}], {scale}"
            )
    return "\n".join(lines)


class ReportGenerator:
    """試行履歴から Markdown レポートを生成するクラス。

    generate_intermediate() は LLM を使わず即座にレポートを生成する。
    generate_final() は LLM を呼び出して AI 考察セクションを追加する。
    """

    def generate_intermediate(
        self,
        trial_records: list[TrialRecord],
        best_params: dict[str, Any],
        best_score: float,
        seed: int | None = None,
        tool_reasoning: str = "",
        current_tool_records: list[TrialRecord] | None = None,
        title: str = "# HPO 中間レポート",
        latest_space_change: SearchSpaceChangeRecord | None = None,
    ) -> str:
        """中間レポートを生成する（LLM 不要）。

        Args:
            trial_records: これまでの試行履歴。
            best_params: 現時点の最良パラメータ。
            best_score: 現時点の最良スコア。
            seed: 乱数シード。
            tool_reasoning: Supervisor が今回のツールを選択した理由。
            current_tool_records: 今回のツール実行で得た TrialRecord リスト。
            title: レポートの見出し（最終レポートでは "# HPO 最終レポート" を渡す）。
            latest_space_change: 今回の実行で行われた探索空間変更。指定時は変更通知セクションを追加。

        Returns:
            Markdown 形式のレポート文字列。
        """
        seed_str = str(seed) if seed is not None else "未指定"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ツール使用内訳
        tool_counts = Counter(r.tool_used for r in trial_records)
        tool_summary = "\n".join(
            f"| {tool} | {count} |" for tool, count in tool_counts.items()
        )

        # スコア推移テーブル（最新10件）
        recent = trial_records[-10:]
        score_rows = "\n".join(
            f"| {r.trial_id} | {r.score:.6f} | {r.tool_used} |" for r in recent
        )

        lines = [
            title,
            "",
            "## メタ情報",
            f"- 実行日時: {now}",
            f"- シード: {seed_str}",
            f"- 現時点の試行回数: {len(trial_records)}",
            "",
            "## 現時点の最良結果",
            f"- スコア: {best_score:.6f}",
            f"- パラメータ: {best_params}",
            "",
            "## スコア推移（直近10件）",
            "| trial_id | score | tool_used |",
            "|----------|-------|-----------|",
            score_rows,
            "",
            "## ツール使用内訳",
            "| ツール名 | 使用回数 |",
            "|---------|---------|",
            tool_summary,
        ]

        # 探索空間の変更通知
        if latest_space_change is not None:
            old_desc = _format_param_space(latest_space_change.old_param_space)
            new_desc = _format_param_space(latest_space_change.new_param_space)
            lines += [
                "",
                "## 探索空間の変更通知",
                f"- 変更時点の試行数: {latest_space_change.trial_id_at_change}",
                "",
                "**変更前:**",
                old_desc,
                "",
                "**変更後:**",
                new_desc,
            ]

        # Supervisor のツール選択理由
        if tool_reasoning:
            lines += [
                "",
                "## Supervisor のツール選択理由",
                tool_reasoning,
            ]

        # ExpertAgentTool のパラメーター提案理由
        if current_tool_records:
            expert_reasonings = [
                r
                for r in current_tool_records
                if r.tool_used == "expert_agent" and r.reasoning
            ]
            if expert_reasonings:
                lines += ["", "## ExpertAgentTool のパラメーター提案理由"]
                for r in expert_reasonings:
                    lines.append(f"- Trial {r.trial_id}: {r.reasoning}")

        # 今回のツール実行で消費した合計時間
        if current_tool_records:
            total_eval = sum(r.eval_duration for r in current_tool_records)
            total_algo = sum(r.algo_duration for r in current_tool_records)
            tool_name = current_tool_records[0].tool_used
            lines += [
                "",
                "## 今回のツール実行時間",
                f"- ツール: {tool_name} / 試行数: {len(current_tool_records)}",
                f"- モデル評価時間（合計）: {total_eval:.4f} 秒",
                f"- アルゴリズム計算時間（合計）: {total_algo:.4f} 秒",
                f"- 合計時間: {total_eval + total_algo:.4f} 秒",
            ]

        return "\n".join(lines)

    def generate_final(
        self,
        trial_records: list[TrialRecord],
        best_params: dict[str, Any],
        best_score: float,
        llm: Any,
        seed: int | None = None,
        generated_param_space: ParamSpace | None = None,
        search_space_change_history: list[SearchSpaceChangeRecord] | None = None,
    ) -> str:
        """最終レポートを生成する（LLM による AI 考察を含む）。

        Args:
            trial_records: 全試行履歴。
            best_params: 最良パラメータ。
            best_score: 最良スコア。
            llm: LLM インスタンス（AI 考察用）。
            seed: 乱数シード。
            generated_param_space: LLM が自動生成したパラメータ空間。指定時はレポートに記載。
            search_space_change_history: 探索空間変更履歴。指定時は変更履歴セクションを追加。

        Returns:
            Markdown 形式のレポート文字列。
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        base = self.generate_intermediate(
            trial_records=trial_records,
            best_params=best_params,
            best_score=best_score,
            seed=seed,
            title="# HPO 最終レポート",
        )

        # 探索空間の変更履歴セクション
        change_history_lines: list[str] = []
        if search_space_change_history:
            change_history_lines += [
                "",
                "## 探索空間の変更履歴",
                "",
                f"{len(search_space_change_history)} 回の変更がありました。",
            ]
            for i, record in enumerate(search_space_change_history, start=1):
                ts = record.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                change_history_lines += [
                    "",
                    f"### 変更 {i}（試行 {record.trial_id_at_change} 件完了後 / {ts}）",
                    "",
                    "**変更前:**",
                    _format_param_space(record.old_param_space),
                    "",
                    "**変更後:**",
                    _format_param_space(record.new_param_space),
                ]

        # LLM が自動生成したパラメータ空間のセクション
        generated_space_lines: list[str] = []
        if generated_param_space is not None:
            generated_space_lines += [
                "",
                "## LLM が自動生成したパラメータ空間",
                "",
                "param_space が未指定のため、LLM がモデル情報をもとに以下の探索空間を設計しました。",
                "",
            ]
            for spec in generated_param_space.specs:
                if spec.type == "categorical":
                    generated_space_lines.append(
                        f"- **{spec.name}**: categorical,"
                        f" choices={list(spec.choices or [])}"
                    )
                else:
                    scale = "log スケール" if spec.log else "linear スケール"
                    generated_space_lines.append(
                        f"- **{spec.name}**: {spec.type},"
                        f" [{spec.low}, {spec.high}], {scale}"
                    )

        # 各ステップでの AI 判断理由の記録
        reasoning_lines: list[str] = []
        reasoning_records = [r for r in trial_records if r.reasoning]
        if reasoning_records:
            reasoning_lines += ["", "## AI 判断理由の記録", ""]
            for r in reasoning_records:
                reasoning_lines.append(
                    f"- **Trial {r.trial_id}** ({r.tool_used}): {r.reasoning}"
                )

        # 全試行の時間サマリー
        timing_lines: list[str] = ["", "## 全試行の時間サマリー"]
        if trial_records:
            total_time = sum(r.eval_duration + r.algo_duration for r in trial_records)
            timing_lines.append(f"- 総最適化時間: {total_time:.4f} 秒")
            timing_lines += [
                "",
                "### ツール別平均",
                "| ツール名 | 平均モデル評価時間（秒） | 平均アルゴリズム計算時間（秒） |",
                "|---------|------------|------------|",
            ]
            for tool in sorted(set(r.tool_used for r in trial_records)):
                tool_records = [r for r in trial_records if r.tool_used == tool]
                avg_eval = sum(r.eval_duration for r in tool_records) / len(
                    tool_records
                )
                avg_algo = sum(r.algo_duration for r in tool_records) / len(
                    tool_records
                )
                timing_lines.append(f"| {tool} | {avg_eval:.4f} | {avg_algo:.4f} |")
        else:
            timing_lines.append("- 試行なし")

        full_report = (
            base
            + "\n".join(change_history_lines)
            + "\n".join(generated_space_lines)
            + "\n".join(reasoning_lines)
            + "\n".join(timing_lines)
        )

        # AI 考察（LLM 生成）
        prompt = (
            f"以下の HPO 実行レポートを読んで、最適化の傾向・結果の解釈・改善提案を "
            f"Markdown の `## AI考察` セクションとして日本語で記述してください。\n\n"
            f"{full_report}"
        )
        response = llm.invoke(
            [
                SystemMessage(content="あなたは機械学習の専門家です。"),
                HumanMessage(content=prompt),
            ]
        )
        ai_content = response.content
        if isinstance(ai_content, list):
            ai_content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in ai_content
            )
        ai_section = f"\n\n{ai_content}"
        return full_report + ai_section
