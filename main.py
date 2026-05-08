# import os
# import openai

# """
# Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

# """

# def call_model(prompt: str, max_tokens=3000, temperature=0.1) -> str:
#     openai.api_key = os.getenv("OPENAI_API_KEY") # please use your own openai api key here.
#     resp = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}],
#         stream=False,
#         max_tokens=max_tokens,
#         temperature=temperature,
#     )
#     return resp.choices[0].message["content"]  # type: ignore

# example_requests = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."


# def main():
#     user_input = input("What kind of story do you want to hear? ")
#     response = call_model(user_input)
#     print(response)


# if __name__ == "__main__":
#     main()


import os
import json
import time
import openai
from dotenv import load_dotenv
"""
Before submitting the assignment, describe here in a few sentences what you would have built next
if you spent 2 more hours on this project:

1. Streaming output — pipe the storyteller's tokens through stdout word-by-word so the story
   "types itself" onto the screen, making it feel magical for a child watching over a parent's shoulder.

2. TTS delivery — after the judge approves the story, call openai.audio.speech.create() with the
   "nova" voice (warm and gentle) so the story is literally read aloud, turning this into a
   hands-free bedtime tool.

3. Session memory — persist the child's name, age, and favourite characters to a local JSON file
   so subsequent runs can offer a sequel ("Want to hear what happened to Alice and Bob next?").
"""


load_dotenv() 


MAX_JUDGE_RETRIES = 3  
MAX_API_RETRIES   = 3     
API_BACKOFF_BASE  = 2      
JUDGE_PASS_AVG    = 3.8    
JUDGE_PASS_SAFETY = 4   
WORD_TARGETS = {
    "quick": (150, 220),
    "full":  (380, 480),
}

_client = None

def get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def call_model(messages: list, max_tokens: int = 800, temperature: float = 0.7) -> str:
    client = get_client()

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content  # note: no ["content"] dict access

        except openai.RateLimitError:
            wait = API_BACKOFF_BASE ** attempt
            print(f"   Rate limit hit — waiting {wait}s (attempt {attempt}/{MAX_API_RETRIES})")
            time.sleep(wait)

        except openai.APIConnectionError:
            wait = API_BACKOFF_BASE ** attempt
            print(f"   Connection error — waiting {wait}s (attempt {attempt}/{MAX_API_RETRIES})")
            time.sleep(wait)

        except openai.APITimeoutError:
            wait = API_BACKOFF_BASE ** attempt
            print(f"   Request timed out — waiting {wait}s (attempt {attempt}/{MAX_API_RETRIES})")
            time.sleep(wait)

        except openai.OpenAIError as e:
            raise RuntimeError(f"OpenAI error: {e}") from e

    raise RuntimeError("OpenAI API unavailable after max retries — please try again later.")


def llm(system: str, user: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
    """Convenience wrapper: system + user message."""
    return call_model(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ── step 1: gather story context from user ────────────────────────────────────

def gather_context() -> dict:
    """Ask the user a few quick questions to personalise the story."""
    print("\nBedtime Story Generator")
    print("─" * 44)

    request = input("\nWhat kind of story would you like?\n> ").strip()
    if not request:
        request = "A girl named Alice and her best friend Bob, who happens to be a cat."
        print(f"(Using example: {request})")

    child_name = input("\nChild's name (press Enter to skip): ").strip()

    age_raw = input("Child's age, 5–10 (press Enter to skip): ").strip()
    age = None
    if age_raw.isdigit():
        age = max(5, min(10, int(age_raw)))  # clamp to 5-10

    moral = input("Any lesson or theme to weave in? e.g. 'sharing', 'being brave' (Enter to skip): ").strip()

    print("\nStory length:")
    print("  1 – Quick  (~3 min read, ~200 words)")
    print("  2 – Full   (~6 min read, ~430 words)")
    length_raw = input("Choose 1 or 2 (default 2): ").strip()
    length_mode = "quick" if length_raw == "1" else "full"

    print("\nTone:")
    print("  1 – Calm & cozy   2 – Exciting   3 – Funny   4 – Magical")
    tone_map = {"1": "calm and cozy", "2": "exciting", "3": "funny", "4": "magical and wondrous"}
    tone = tone_map.get(input("Choose 1–4 (default 1): ").strip(), "calm and cozy")

    return {
        "request":     request,
        "child_name":  child_name,
        "age":         age,
        "moral":       moral,
        "length_mode": length_mode,
        "tone":        tone,
    }


# ── step 2: classify & extract themes (open-ended) ───────────────────────────

CLASSIFIER_SYSTEM = """You are a children's story librarian.
Given a story request, extract the key storytelling attributes.

Reply with ONLY valid JSON — no markdown, no extra text:
{
  "genre":      "<one of: adventure | friendship | animal | fantasy | mystery | comedy | nature | general>",
  "setting":    "<brief setting description, e.g. 'enchanted forest' or 'outer space'>",
  "characters": "<main characters, e.g. 'a brave girl and her cat'>",
  "mood":       "<overall mood, e.g. 'playful' or 'heartwarming'>",
  "themes":     ["<theme1>", "<theme2>"]
}"""

def classify(request: str) -> dict:
    """Extracts genre, setting, characters, mood, themes from the free-text request."""
    raw = llm(CLASSIFIER_SYSTEM, request, max_tokens=200, temperature=0.0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "genre": "general", "setting": "a magical world",
            "characters": "a young hero", "mood": "warm", "themes": ["friendship"],
        }


# ── step 3: build the storyteller prompt ──────────────────────────────────────

STORYTELLER_SYSTEM = """You are a beloved children's author with a gift for vivid, memorable stories.

Your narrative style:
- Open with a sensory hook that drops the reader straight into the world (a sound, a smell, a feeling).
- Vary sentence rhythm: short punchy sentences for action, longer flowing ones for calm moments.
- Use one recurring detail from Act 1 as a callback in Act 3 — it makes stories feel complete.
- Give every character one distinct quirk or habit that makes them feel real.
- End on a single warm image that lingers in the mind.

Story structure — three acts:
  Act 1 (Setup):      Introduce character(s) and world. Plant the seed of the problem.
  Act 2 (Problem):    The challenge unfolds. Show effort, setback, then a creative solution.
  Act 3 (Resolution): Resolve warmly. Echo Act 1. Leave the reader smiling.

Non-negotiable rules:
- Vocabulary and concepts appropriate for the specified age.
- No violence, scary content, adult themes, or sad/ambiguous endings.
- The story MUST end happily or hopefully.
- Write in third-person past tense.
- Paragraphs: 2–4 sentences each."""

def build_story_prompt(ctx: dict, attrs: dict, critique: str = "") -> str:
    word_min, word_max = WORD_TARGETS[ctx["length_mode"]]
    age_str    = f"age {ctx['age']}" if ctx["age"] else "ages 5–10"
    name_str   = (f"Weave the name '{ctx['child_name']}' in as a minor character or a "
                  f"narrator aside so it feels personal.") if ctx["child_name"] else ""
    moral_str  = (f"Subtly bake in this lesson through action, not lecture: "
                  f"'{ctx['moral']}'.") if ctx["moral"] else ""
    critique_str = (f"\n\nPrevious draft failed review. Fix these specific issues:\n{critique}"
                    if critique else "")

    return f"""Write a {ctx['tone']} children's story for {age_str}.

Story request: "{ctx['request']}"

Extracted story attributes:
- Genre:      {attrs.get('genre', 'general')}
- Setting:    {attrs.get('setting', 'a magical world')}
- Characters: {attrs.get('characters', 'a young hero')}
- Mood:       {attrs.get('mood', 'warm')}
- Themes:     {', '.join(attrs.get('themes', ['friendship']))}

{name_str}
{moral_str}

Target length: {word_min}–{word_max} words.
{critique_str}

Write the story now."""


# ── step 4: age-appropriateness pre-screen (hard binary gate) ─────────────────

PRESCREEN_SYSTEM = """You are a child safety reviewer for a children's book publisher.
Check the story ONLY for hard violations.

Reply with ONLY valid JSON — no markdown, no extra text:
{
  "safe":   <true|false>,
  "reason": "<empty string if safe, one sentence naming the violation if not>"
}

Hard violations (any one = not safe):
- Vocabulary a 10-year-old would not understand (without context clues to help)
- Concepts beyond ages 5–10: death, romance, war, politics, trauma, addiction
- Scary imagery used to threaten (monsters intending real harm, dangerous darkness)
- Unhappy or ambiguous ending"""

def prescreen(story: str) -> tuple:
    """Returns (is_safe: bool, reason: str). Fails open on parse error."""
    raw = llm(PRESCREEN_SYSTEM, story, max_tokens=100, temperature=0.0)
    try:
        result = json.loads(raw)
        return result.get("safe", True), result.get("reason", "")
    except json.JSONDecodeError:
        return True, ""


# ── step 5: LLM quality judge ─────────────────────────────────────────────────

JUDGE_SYSTEM = """You are a strict but fair editor for a children's book publisher.
Evaluate the story on four dimensions, each scored 1–5.

1. age_appropriateness – vocabulary and concepts right for the target age
2. story_arc           – clear setup, problem, and satisfying resolution present
3. engagement          – vivid details, interesting characters, emotional pull
4. safety              – no scary content, violence, adult themes, or unhappy endings

Reply with ONLY valid JSON — no markdown, no extra text:
{
  "scores": {
    "age_appropriateness": <int 1-5>,
    "story_arc":           <int 1-5>,
    "engagement":          <int 1-5>,
    "safety":              <int 1-5>
  },
  "average": <float>,
  "pass":    <true|false>,
  "critique": "<one sentence per failing dimension; empty string if all pass>"
}

Pass = average >= 3.8 AND safety >= 4.
If safety < 4, always set pass to false regardless of average."""

def judge(story: str) -> dict:
    """Returns parsed judge dict. Fails open on parse error."""
    raw = llm(JUDGE_SYSTEM, f"Evaluate this story:\n\n{story}", max_tokens=300, temperature=0.2)
    for candidate in [raw, raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    # Fail open
    return {"scores": {}, "average": 4.0, "pass": True, "critique": ""}


# ── core pipeline ─────────────────────────────────────────────────────────────

def generate_story(ctx: dict) -> str:
    """classify → build prompt → write → prescreen → judge → (retry) → return best story"""
    print("\nAnalysing your request...", flush=True)
    attrs = classify(ctx["request"])
    print(f"   Genre: {attrs.get('genre')} | Setting: {attrs.get('setting')} | "
          f"Themes: {', '.join(attrs.get('themes', []))}", flush=True)

    critique   = ""
    best_story = ""
    best_avg   = -1.0

    for attempt in range(1, MAX_JUDGE_RETRIES + 1):
        print(f"\nWriting story (attempt {attempt}/{MAX_JUDGE_RETRIES})...", flush=True)
        story = llm(STORYTELLER_SYSTEM, build_story_prompt(ctx, attrs, critique),
                    max_tokens=700, temperature=0.85)

        # Hard gate first — fast, cheap, binary
        safe, reason = prescreen(story)
        if not safe:
            print(f"   Pre-screen failed: {reason}", flush=True)
            critique = f"Safety pre-screen failed: {reason}. Rewrite to remove this issue."
            best_story = best_story or story  # keep as last-resort fallback
            continue

        # Quality scoring
        print("Judge reviewing...", flush=True)
        verdict = judge(story)
        avg     = verdict.get("average", 0)
        passed  = verdict.get("pass", False)
        scores  = verdict.get("scores", {})

        print(
            f"   age:{scores.get('age_appropriateness','?')} "
            f"arc:{scores.get('story_arc','?')} "
            f"engagement:{scores.get('engagement','?')} "
            f"safety:{scores.get('safety','?')} "
            f"| avg:{avg:.1f} | {'PASS' if passed else 'FAIL'}",
            flush=True,
        )

        if avg > best_avg:
            best_avg   = avg
            best_story = story

        if passed:
            break

        critique = verdict.get("critique", "")
        if critique:
            print(f"   Critique: {critique}", flush=True)

    return best_story


# ── feedback loop ─────────────────────────────────────────────────────────────

def run():
    ctx = gather_context()

    while True:
        story = generate_story(ctx)

        print("\n" + "─" * 60)
        print(story)
        print("─" * 60 + "\n")

        print("Would you like any changes?")
        print("  e.g. 'make it funnier', 'add a dragon', 'make it shorter'")
        print("  Or press Enter to finish.\n")
        feedback = input("Your feedback: ").strip()

        if not feedback:
            print("\nGoodnight.\n")
            break

        # Merge feedback into the original request and loop
        ctx["request"] = f"{ctx['request']} — also: {feedback}"


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()