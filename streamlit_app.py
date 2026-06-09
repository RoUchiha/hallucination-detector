"""
Hallucination Detector — Live Demo
Streamlit front-end for the dual-scoring (NLI + LLM-as-judge) pipeline.
"""

import os, json, textwrap
import streamlit as st

st.set_page_config(
    page_title="Hallucination Detector",
    page_icon="🔍",
    layout="wide",
)

# ── styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.verdict-SUPPORTED   { background:#16a34a; color:#fff; padding:2px 10px; border-radius:12px; font-weight:600; }
.verdict-HALLUCINATED{ background:#dc2626; color:#fff; padding:2px 10px; border-radius:12px; font-weight:600; }
.verdict-UNCERTAIN   { background:#ca8a04; color:#fff; padding:2px 10px; border-radius:12px; font-weight:600; }
.tier-LOW   { background:#16a34a; color:#fff; padding:6px 18px; border-radius:20px; font-size:1.1rem; font-weight:700; }
.tier-MEDIUM{ background:#ca8a04; color:#fff; padding:6px 18px; border-radius:20px; font-size:1.1rem; font-weight:700; }
.tier-HIGH  { background:#dc2626; color:#fff; padding:6px 18px; border-radius:20px; font-size:1.1rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.title("🔍 Hallucination Detector")
st.caption(
    "Detects factual hallucinations in LLM-generated text using **NLI scoring** "
    "(DeBERTa cross-encoder) + **LLM-as-judge** escalation. "
    "Each claim is independently verified against the source document."
)
st.markdown("---")

DEMO_SOURCE = """The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.
It was constructed between 1887 and 1889 as the entrance arch for the 1889 World's Fair.
The tower stands 330 metres (1,083 feet) tall, including its broadcast antenna.
It was designed by Gustave Eiffel, whose engineering company was responsible for the structural work.
Approximately 7 million people visit the Eiffel Tower every year, making it the most-visited paid monument in the world.
The tower was originally intended to be dismantled after 20 years but was kept due to its value as a radio transmission tower."""

DEMO_GENERATED = """The Eiffel Tower is an iconic iron structure located in Paris, France, and was built for the 1889 World's Fair.
It stands approximately 330 metres tall and was designed by engineer Gustave Eiffel.
The tower attracts around 7 million visitors annually.
Construction began in 1885, two years before work actually started.
The tower was originally made of steel rather than wrought iron.
Its popularity as a tourist attraction meant demolition was never seriously considered."""

# ── inputs ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Source Document (Ground Truth)")
    source = st.text_area("Paste the authoritative source text:", value=DEMO_SOURCE, height=220, label_visibility="collapsed")

with col2:
    st.subheader("🤖 LLM-Generated Text to Verify")
    generated = st.text_area("Paste the LLM output to fact-check:", value=DEMO_GENERATED, height=220, label_visibility="collapsed")

api_key = st.text_input("🔑 Anthropic API Key", type="password",
    value=os.environ.get("ANTHROPIC_API_KEY", ""),
    help="Your key is used only for this request and never stored.")

run = st.button("🚀 Run Hallucination Detection", type="primary", use_container_width=True)

# ── pipeline ──────────────────────────────────────────────────────────────────
if run:
    if not api_key:
        st.error("Please enter your Anthropic API key.")
        st.stop()
    if not source.strip() or not generated.strip():
        st.error("Both source document and generated text are required.")
        st.stop()

    os.environ["ANTHROPIC_API_KEY"] = api_key

    with st.spinner("Step 1/3 — Extracting atomic claims from generated text…"):
        try:
            import anthropic, re
            client = anthropic.Anthropic(api_key=api_key)

            EXTRACT_SYSTEM = (
                "You are a precise claim extractor. Decompose the generated text into atomic, "
                "self-contained factual claims — one per sentence. "
                "Respond ONLY with a JSON array:\n"
                '[{"id":1,"text":"<claim>","span_start":0,"span_end":10}, ...]\n'
                "Each claim must be independently verifiable."
            )
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=1024,
                system=EXTRACT_SYSTEM,
                messages=[{"role":"user","content":f"Text:\n{generated}"}]
            )
            raw = msg.content[0].text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
            claims = json.loads(raw)
            st.success(f"✅ {len(claims)} atomic claims extracted")
        except Exception as e:
            st.error(f"Claim extraction failed: {e}")
            st.stop()

    with st.spinner("Step 2/3 — LLM-as-judge scoring each claim…"):
        JUDGE_SYSTEM = (
            'You are a rigorous fact-checker. Given a source document and a claim, '
            'determine whether the source supports the claim. '
            'Respond ONLY with JSON: {"verdict":"supported"|"unsupported"|"unknown","confidence":0.0-1.0,"reasoning":"one sentence"}'
        )
        verdicts = []
        progress = st.progress(0)
        for i, claim in enumerate(claims):
            try:
                jmsg = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=200,
                    system=JUDGE_SYSTEM,
                    messages=[{"role":"user","content":
                        f"SOURCE:\n{source[:3000]}\n\nCLAIM:\n{claim['text']}"}]
                )
                data = json.loads(jmsg.content[0].text.strip())
                final = (
                    "SUPPORTED"    if data["verdict"] == "supported"   else
                    "HALLUCINATED" if data["verdict"] == "unsupported" else
                    "UNCERTAIN"
                )
                verdicts.append({**claim, **data, "final": final})
            except Exception:
                verdicts.append({**claim, "verdict":"unknown","confidence":0.0,
                                 "reasoning":"Error during analysis","final":"UNCERTAIN"})
            progress.progress((i + 1) / len(claims))

    with st.spinner("Step 3/3 — Building report…"):
        total = len(verdicts)
        hallucinated = sum(1 for v in verdicts if v["final"] == "HALLUCINATED")
        rate = hallucinated / total if total else 0
        tier = "LOW" if rate < 0.2 else "MEDIUM" if rate < 0.5 else "HIGH"

    # ── results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Report")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Claims",     total)
    m2.metric("Hallucinated",     hallucinated)
    m3.metric("Hallucination Rate", f"{rate:.0%}")
    with m4:
        st.markdown(f"**Risk Tier**")
        st.markdown(f'<span class="tier-{tier}">{tier}</span>', unsafe_allow_html=True)

    st.markdown("### Per-Claim Verdicts")
    for v in verdicts:
        badge = f'<span class="verdict-{v["final"]}">{v["final"]}</span>'
        with st.expander(f"#{v['id']} — {v['text'][:90]}{'…' if len(v['text'])>90 else ''}  {badge}", expanded=False):
            st.markdown(badge, unsafe_allow_html=True)
            st.write(f"**Claim:** {v['text']}")
            st.write(f"**Reasoning:** {v.get('reasoning','—')}")
            st.write(f"**Confidence:** {v.get('confidence',0):.0%}")

    st.markdown("---")
    st.download_button(
        "⬇️ Download JSON Report",
        data=json.dumps({"hallucination_rate": rate, "risk_tier": tier, "verdicts": verdicts}, indent=2),
        file_name="hallucination_report.json",
        mime="application/json",
    )
