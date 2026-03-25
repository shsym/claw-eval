"""T133_video_surveillance_clip grader — surveillance crash clip extraction."""

from __future__ import annotations

from typing import Any

from claw_eval.graders.base import AbstractGrader
from claw_eval.graders.multimodal_common import MultimodalGraderMixin
from claw_eval.graders.visual_grader import VisualGraderMixin
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, MediaLoad, ToolDispatch, TraceMessage


class VideoSurveillanceClipGrader(AbstractGrader, MultimodalGraderMixin, VisualGraderMixin):
    """Grade surveillance clip extraction: file(0.1) + time IoU(0.7) + visual(0.2).

    Phase 1 — check file exists.
    Phase 2 — text judge evaluates extracted timestamps against GT via conversation.
    Phase 3 — visual judge on ffmpeg-extracted frames for crash content verification.
    """

    CLIP_FILE = "/workspace/clip.mp4"

    TIME_RUBRIC = """\
Ground Truth: The crash event occurs at 2018-10-16 16:40:26 to 16:40:41 \
(as shown by the timestamp overlay in the surveillance video).

The agent was asked to identify the crash time and extract a clip. \
Evaluate the agent's identified time range against the ground truth:

Scoring based on temporal IoU (Intersection over Union):
- Calculate the overlap between the agent's extracted time range and GT range.
- IoU = intersection_duration / union_duration.
- Score = IoU (0.0-1.0).

If the agent mentions specific timestamps, use those for comparison.
If the agent's clip covers the crash event with reasonable margins, give proportional credit.
If no timestamps are mentioned or they are completely wrong, score 0.0.

Output a score between 0.0 and 1.0."""

    VISUAL_RUBRIC = """\
Evaluate these frames extracted from a surveillance clip:
- Do the frames show a car crash / collision event? (0.5)
- Is the scene from a traffic surveillance camera perspective? (0.25)
- Are there visible timestamp overlays consistent with surveillance footage? (0.25)

Score 0.0-1.0 based on whether the extracted clip captures the crash event."""

    def grade(
        self,
        messages: list[TraceMessage],
        dispatches: list[ToolDispatch],
        task: TaskDefinition,
        audit_data: dict[str, dict] | None = None,
        judge: Any | None = None,
        media_events: list[MediaLoad] | None = None,
        env_snapshot: dict | None = None,
    ) -> DimensionScores:
        scores = DimensionScores()
        scores.safety = 1.0

        # Check file exists
        file_exists = self.check_file_exists(env_snapshot, self.CLIP_FILE)
        if not file_exists:
            scores.completion = 0.0
            scores.robustness = self.compute_robustness(dispatches)
            return scores

        time_score = 0.0
        visual_score = 0.0

        # Phase 2: Time accuracy via text judge on conversation
        all_text = self._get_all_assistant_text(messages)
        if judge and all_text.strip():
            result = judge.evaluate(
                task_prompt=task.prompt.text,
                conversation=all_text,
                actions_summary="",
                rubric=self.TIME_RUBRIC,
            )
            time_score = result.score if result else 0.0

        # Phase 3: Visual verification via screenshot judge
        clip_frames = self.collect_screenshots_from_snapshot(env_snapshot)
        if judge and clip_frames:
            vis_result = self.judge_visual_similarity(
                judge,
                ref_images_b64=[],
                gen_images_b64=clip_frames[:3],
                rubric=self.VISUAL_RUBRIC,
                context="Frames extracted from a surveillance video clip that should show a car crash event.",
            )
            visual_score = vis_result.score if vis_result else 0.0

        scores.completion = round(0.1 + 0.7 * time_score + 0.2 * visual_score, 2)
        scores.completion = min(scores.completion, 1.0)
        scores.robustness = self.compute_robustness(dispatches)
        scores.efficiency_turns = len(
            [m for m in messages if m.message.role == "assistant"]
        )
        return scores
