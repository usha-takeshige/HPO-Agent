"""Report generation for HPO runs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from hpo_agent.models import TrialRecord


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
    ) -> str:
        """中間レポートを生成する（LLM 不要）。

        Args:
            trial_records: これまでの試行履歴。
            best_params: 現時点の最良パラメータ。
            best_score: 現時点の最良スコア。
            seed: 乱数シード。

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

        # 実行時間サマリー
        if trial_records:
            avg_eval = sum(r.eval_duration for r in trial_records) / len(trial_records)
            avg_algo = sum(r.algo_duration for r in trial_records) / len(trial_records)
            timing_summary = (
                f"- 評価時間（平均）: {avg_eval:.4f} 秒\n"
                f"- アルゴリズム時間（平均）: {avg_algo:.4f} 秒"
            )
        else:
            timing_summary = "- 試行なし"

        lines = [
            "# HPO 中間レポート",
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
            "",
            "## 実行時間サマリー",
            timing_summary,
        ]
        return "\n".join(lines)

    def generate_final(
        self,
        trial_records: list[TrialRecord],
        best_params: dict[str, Any],
        best_score: float,
        llm: Any,
        seed: int | None = None,
    ) -> str:
        """最終レポートを生成する（LLM による AI 考察を含む）。

        Args:
            trial_records: 全試行履歴。
            best_params: 最良パラメータ。
            best_score: 最良スコア。
            llm: LLM インスタンス（AI 考察用）。
            seed: 乱数シード。

        Returns:
            Markdown 形式のレポート文字列。
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        intermediate = self.generate_intermediate(
            trial_records=trial_records,
            best_params=best_params,
            best_score=best_score,
            seed=seed,
        )

        prompt = (
            f"以下の HPO 実行レポートを読んで、最適化の傾向・結果の解釈・改善提案を "
            f"Markdown の `## AI考察` セクションとして日本語で記述してください。\n\n"
            f"{intermediate}"
        )
        response = llm.invoke(
            [
                SystemMessage(content="あなたは機械学習の専門家です。"),
                HumanMessage(content=prompt),
            ]
        )
        ai_section = f"\n\n## AI考察\n\n{response.content}"
        return intermediate + ai_section
