"""What the model is holding: the live-image budget, elision, and compaction.

Images dominate context cost. One 2000x1333 render is about 3,400 visual tokens,
so twenty of them cost more than the rest of a long session put together. A loop
that simply appends every image it takes dies around step thirty, and it dies by
silently truncating rather than by raising.

So this keeps a fixed number of images live and swaps older ones for a text stub
naming the artifact that made them (README.md:91). The stub is only honest
because `view("a_0042")` brings the image back — see `tools.resolve_ref`.

Messages are provider-neutral. Blocks are plain dicts with a `type`, and a
backend maps them to whatever its API wants; nothing here imports a vendor SDK.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

from PIL import Image

from visru.config import Config
from visru.geometry import visual_tokens
from visru.tools import ToolResult

# Rough, and deliberately so: a real tokeniser call per step would cost a network
# round trip to learn something the budget only needs to the nearest few percent.
CHARS_PER_TOKEN = 4
STUB_CHARS = 140


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolOutput:
    """One tool result, and whether its images are still live."""

    call_id: str
    name: str
    text: str
    images: list[Image.Image] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    elided: bool = False

    @property
    def live_images(self) -> int:
        return 0 if self.elided else len(self.images)

    def stub(self) -> str:
        """The note that replaces the pixels.

        Keeps the first line of the tool's own text, which already names the asset,
        the box and the scale — the three things the model needs to decide whether
        this image is worth asking back.
        """
        head = self.text.splitlines()[0] if self.text else ""
        for artifact_id in self.artifacts:  # the ID is repeated by the wrapper
            head = head.replace(f"artifact={artifact_id}", "").strip()
        if len(head) > STUB_CHARS:
            head = head[:STUB_CHARS].rstrip() + "…"
        named = ", ".join(self.artifacts) or self.name
        return f"[{named} elided: {head}]"

    def elide(self) -> int:
        """Drop the pixels, keep the record. Returns how many images went."""
        freed = self.live_images
        self.elided = True
        return freed


@dataclass
class Step:
    """One model turn and the tool results it asked for.

    Compaction works in whole steps because the providers require every tool call
    to be answered; folding away half a step would leave an unanswered call and
    the request would be rejected.
    """

    text: str
    calls: list[ToolCall] = field(default_factory=list)
    outputs: list[ToolOutput] = field(default_factory=list)


class Context:
    def __init__(self, task: str, cfg: Config, system: str = ""):
        self.task = task
        self.cfg = cfg
        self.system = system
        self.steps: list[Step] = []
        self.summary: str = ""
        self.elisions = 0
        self.compactions = 0

    # --- building ---------------------------------------------------------
    def add_turn(self, text: str, calls: list[ToolCall] | None = None) -> Step:
        step = Step(text=text, calls=list(calls or []))
        self.steps.append(step)
        return step

    def add_output(self, call: ToolCall, result: ToolResult) -> ToolOutput:
        if not self.steps:
            raise ValueError("a tool result needs a model turn to answer")
        out = ToolOutput(call_id=call.id, name=call.name, text=result.text,
                         images=list(result.images), artifacts=list(result.artifacts))
        self.steps[-1].outputs.append(out)
        self.enforce()
        return out

    def add_error(self, call: ToolCall, message: str) -> ToolOutput:
        """A failed call still has to answer its tool_use, or the next request 400s."""
        return self.add_output(call, ToolResult(text=f"error: {message}"))

    # --- budget -----------------------------------------------------------
    @property
    def live_images(self) -> int:
        return sum(o.live_images for step in self.steps for o in step.outputs)

    def _outputs(self) -> Iterator[ToolOutput]:
        for step in self.steps:
            yield from step.outputs

    def estimate_tokens(self) -> int:
        patch = self.cfg.model.patch
        chars = len(self.system) + len(self.task) + len(self.summary)
        tokens = 0
        for step in self.steps:
            chars += len(step.text)
            for call in step.calls:
                chars += len(call.name) + len(json.dumps(call.arguments, default=str))
            for out in step.outputs:
                chars += len(out.stub()) if out.elided else len(out.text)
                if not out.elided:
                    tokens += sum(visual_tokens(im.width, im.height, patch)
                                  for im in out.images)
        return tokens + chars // CHARS_PER_TOKEN

    # --- keeping it inside the limits -------------------------------------
    def enforce(self) -> None:
        self._elide()
        self._compact()

    def _elide(self) -> None:
        """Swap the oldest images for stubs once the budget is exceeded.

        A batch at a time, not one per step: rewriting history invalidates the
        cached prefix, so doing it every time an image arrives would pay the
        cache miss on every single turn (config.py:34).
        """
        limit = self.cfg.context.max_live_images
        if self.live_images <= limit:
            return
        freed = 0
        for out in self._outputs():  # oldest first
            if freed >= self.cfg.context.elide_batch:
                break
            if out.live_images:
                freed += out.elide()
                self.elisions += 1

    def _compact(self) -> None:
        """Fold the oldest steps into one note when the whole thing grows too big.

        Artifact IDs survive verbatim. A compacted summary that dropped them would
        strand the evidence: the model would cite an ID the ledger can still see
        but the model can no longer name, and the claim would be refused for a
        reason invisible to it.
        """
        if self.estimate_tokens() <= self.cfg.context.compact_at_tokens:
            return
        if len(self.steps) < 4:  # nothing to fold that is not still in play
            return

        keep = max(2, len(self.steps) // 2)
        folded, self.steps = self.steps[:-keep], self.steps[-keep:]

        lines, seen = [], []
        for step in folded:
            head = step.text.strip().splitlines()[0] if step.text.strip() else ""
            if head:
                lines.append(f"- {head[:STUB_CHARS]}")
            for out in step.outputs:
                for artifact_id in out.artifacts:
                    if artifact_id not in seen:
                        seen.append(artifact_id)

        self.summary = (
            f"{self.summary}\n" if self.summary else "") + (
            f"[{len(folded)} earlier steps compacted]\n" + "\n".join(lines) +
            (f"\nArtifacts from those steps, still viewable by ID: {', '.join(seen)}"
             if seen else ""))
        self.compactions += 1

    # --- rendering --------------------------------------------------------
    def render(self) -> list[dict[str, Any]]:
        """Provider-neutral messages. Blocks carry PIL images; backends encode them."""
        opening = self.task if not self.summary else f"{self.task}\n\n{self.summary}"
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": opening}]}]

        for step in self.steps:
            assistant: list[dict[str, Any]] = []
            if step.text:
                assistant.append({"type": "text", "text": step.text})
            for call in step.calls:
                assistant.append({"type": "tool_use", "id": call.id,
                                  "name": call.name, "input": call.arguments})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})

            if step.outputs:
                results = [{"type": "tool_result", "tool_use_id": out.call_id,
                            "content": self._result_blocks(out)}
                           for out in step.outputs]
                messages.append({"role": "user", "content": results})
        return messages

    @staticmethod
    def _result_blocks(out: ToolOutput) -> list[dict[str, Any]]:
        """Images first, then text — the order the MCP server already established."""
        if out.elided:
            return [{"type": "text", "text": out.stub()}]
        blocks: list[dict[str, Any]] = [{"type": "image", "image": im}
                                        for im in out.images]
        blocks.append({"type": "text", "text": out.text})
        return blocks
