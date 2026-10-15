import streamlit as st
import re
import hashlib
import json
import os
import pypdf
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="CIAT25 PDP Sovereign Interceptor", layout="wide")

st.title("CIAT25 Sovereign Core: PDP Law & Vector Governance Engine")

# Load SentenceTransformer model with Streamlit caching
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# --- Native Anchor Standard & PDP Rules Loader ---
def load_anchor_standard() -> tuple[list[str], list[str]]:
    schema_path = "schemas/anchor_standard.json"
    corpus = []
    prohibited = [
        "guaranteed cure", "100% effective", "miracle treatment", 
        "instant elimination", "unencrypted storage", "bypass consent"
    ]
    
    if os.path.exists(schema_path):
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    corpus = data.get("anchor_corpus", [])
                    prohibited = data.get("prohibited_assertions", prohibited)
        except Exception as e:
            st.sidebar.warning(f"Failed to parse anchor_standard.json: {e}")
            
    if not corpus:
        corpus = [
            "All data processing must comply with local Personal Data Protection regulations and strict PII anonymization mandates.",
            "Egypt PDP Law No. 151/2020 mandates local data sovereignty, air-gapped execution, and perimeter defense against cross-border data leakage.",
            "Local data retention compliance, internal customer metadata records, and Alexandria edge node processing ensure total sovereign control."
        ]
    return corpus, prohibited

# Initialize Session State
if "telemetry_logs" not in st.session_state:
    st.session_state.telemetry_logs = []

anchor_texts, prohibited_assertions = load_anchor_standard()

if "anchor_embeddings" not in st.session_state:
    st.session_state.anchor_embeddings = [model.encode(chunk, convert_to_numpy=True) for chunk in anchor_texts]

# --- Gate -1: Deterministic PII & Injection Firewall ---
def gate_minus_one(raw_input: str) -> tuple[str, bool, str]:
    if not raw_input or not raw_input.strip():
        return "", False, "Gate -1 Drop: Empty payload."
    
    cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', raw_input)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    if len(cleaned) > 10000:
        return "", False, "Gate -1 Drop: Payload exceeds maximum length bounds (10,000 chars)."
        
    egyptian_id_pattern = r'\b[23]\d{12}\b'
    phone_pattern = r'\b01[0125]\d{8}\b'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    if re.search(egyptian_id_pattern, cleaned):
        return "", False, "Gate -1 PDP Violation: Unmasked Egyptian National ID detected."
    if re.search(phone_pattern, cleaned):
        return "", False, "Gate -1 PDP Violation: Unmasked local mobile telephone number detected."
    if re.search(email_pattern, cleaned):
        return "", False, "Gate -1 PDP Violation: Raw email address detected in inbound payload."
        
    injection_patterns = ["ignore previous instructions", "system override", "jailbreak", "dump database"]
    for pattern in injection_patterns:
        if pattern in cleaned.lower():
            return "", False, f"Gate -1 Drop: Security anomaly matched restricted pattern: '{pattern}'."
            
    return cleaned, True, "Gate -1 Pass: Perimeter PII & structural integrity verified."

# --- Gate 0: Trajectory Sensor (Semantic Cosine Similarity) ---
def gate_zero(query_text: str, anchor_embeddings: list[np.ndarray], threshold: float = 0.20) -> tuple[str, float, str]:
    if not anchor_embeddings:
        return "DROP", 0.0, "Gate 0 Drop: Active anchor corpus is empty."
    
    query_vector = model.encode(query_text, convert_to_numpy=True)
    scores = [
        np.dot(query_vector, a_vec) / (np.linalg.norm(query_vector) * np.linalg.norm(a_vec))
        for a_vec in anchor_embeddings
    ]
    best_score = float(max(scores)) if scores else 0.0
    
    if best_score < threshold:
        return "DROP", best_score, f"Gate 0 Drop: Trajectory alignment ({best_score:.4f}) below threshold ({threshold})."
        
    return "PASS_TO_METABOLIC", best_score, f"Gate 0 Pass: Trajectory aligned. Max similarity: {best_score:.4f}."

# --- Gate 1: Egress PDP & Anchor Confidence Engine ---
class Gate1PDPEnforcementEngine:
    def __init__(self, beta: float = 10.0, min_claim_threshold: float = 0.75):
        self.beta = beta  
        self.min_claim_threshold = min_claim_threshold

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def evaluate_payload(
        self, 
        text: str, 
        claim_embeddings: list[np.ndarray], 
        anchor_embeddings: list[np.ndarray],
        forbidden_list: list[str]
    ) -> dict:
        text_lower = text.lower()
        for forbidden in forbidden_list:
            if forbidden in text_lower:
                return {
                    "confidence_score": 0.0,
                    "verdict": "PDP_BREACH_HARD_STOP",
                    "reason": f"Prohibited regulatory assertion triggered: '{forbidden}'"
                }

        if not claim_embeddings or not anchor_embeddings:
            return {"confidence_score": 0.0, "verdict": "ANCHOR_VOID", "reason": "EMPTY_INPUTS"}

        claim_scores = []
        for c_vec in claim_embeddings:
            best_match = max([self._cosine_similarity(c_vec, a_vec) for a_vec in anchor_embeddings])
            claim_scores.append(best_match)

        claim_scores_arr = np.array(claim_scores)
        soft_min_score = - (1.0 / self.beta) * np.log(
            np.mean(np.exp(-self.beta * claim_scores_arr))
        )

        final_confidence = round(float(soft_min_score), 4)
        
        if final_confidence >= self.min_claim_threshold:
            verdict = "PASS"
            reason = "Passed PDP & confidence bounds."
        else:
            verdict = "ANCHOR_VOID"
            reason = f"Confidence score ({final_confidence}) below minimum required threshold ({self.min_claim_threshold})."

        return {
            "confidence_score": final_confidence,
            "metrics": {"soft_min_cosine": final_confidence},
            "verdict": verdict,
            "reason": reason
        }

# --- Sidebar Controls ---
st.sidebar.header("Sovereign Telemetry Controls")
strict_mode = st.sidebar.checkbox("Enforce Hard Drop Mode", value=True)
g0_threshold = st.sidebar.slider("Gate 0 Trajectory Threshold", min_value=0.05, max_value=0.50, value=0.20, step=0.05)
g1_threshold = st.sidebar.slider("Gate 1 PDP Confidence Threshold", min_value=0.40, max_value=0.90, value=0.75, step=0.01)

if st.sidebar.button("Clear Audit History"):
    st.session_state.telemetry_logs = []
    st.sidebar.success("Audit log wiped.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Inbound Protected Payload")
    user_input = st.text_area("Enter production text, prompt, or model output to audit:", height=150)
    st.caption("Tip: Try pasting text containing an Egyptian National ID (e.g., 29912150101234) or forbidden claims to test hard-stop triggers.")

with col2:
    st.subheader("Sovereign Execution Pipeline")
    if st.button("Execute PDP Pipeline"):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # 1. Gate -1 (PII Firewall)
        clean_text, g1_pass, g1_msg = gate_minus_one(user_input)
        
        if not g1_pass:
            log_entry = {
                "Timestamp": timestamp,
                "Gate -1": "DROP",
                "Gate 0": "N/A",
                "Gate 1 Verdict": "PDP_BREACH_BLOCKED",
                "Score": 0.0,
                "Message": g1_msg
            }
            st.session_state.telemetry_logs.append(log_entry)
            st.error(f"Perimeter Blocked at Gate -1: {g1_msg}")
        else:
            # 2. Gate 0 (Trajectory Sensor)
            action, sim_score, g0_msg = gate_zero(clean_text, st.session_state.anchor_embeddings, threshold=g0_threshold)
            
            if action == "DROP" and strict_mode:
                log_entry = {
                    "Timestamp": timestamp,
                    "Gate -1": "PASS",
                    "Gate 0": "DROP",
                    "Gate 1 Verdict": "N/A",
                    "Score": round(sim_score, 4),
                    "Message": g0_msg
                }
                st.session_state.telemetry_logs.append(log_entry)
                st.error(f"Trajectory Blocked at Gate 0: {g0_msg}")
            else:
                # 3. Gate 1 (PDP Enforcement & Confidence Engine) - Holistic + Sentence Hybrid
                claim_embeddings = [model.encode(clean_text, convert_to_numpy=True)]
                for s in [sent.strip() for sent in clean_text.split('.') if sent.strip()]:
                    if len(s) > 10:
                        claim_embeddings.append(model.encode(s, convert_to_numpy=True))
                
                pdp_engine = Gate1PDPEnforcementEngine(min_claim_threshold=g1_threshold)
                eval_result = pdp_engine.evaluate_payload(
                    clean_text, 
                    claim_embeddings, 
                    st.session_state.anchor_embeddings, 
                    prohibited_assertions
                )
                
                log_id = hashlib.sha256(clean_text.encode()).hexdigest()[:10]
                log_entry = {
                    "Timestamp": timestamp,
                    "Gate -1": "PASS",
                    "Gate 0": action,
                    "Gate 1 Verdict": eval_result.get("verdict"),
                    "Score": eval_result.get("confidence_score", 0.0),
                    "Message": f"ID: {log_id} | {eval_result.get('reason')}"
                }
                st.session_state.telemetry_logs.append(log_entry)
                
                if eval_result.get("verdict") == "PASS":
                    st.success(f"Pipeline Cleared | Log ID: {log_id} | PDP Confidence: {eval_result.get('confidence_score')}")
                else:
                    st.error(f"Gate 1 PDP Hard-Stop Triggered | Log ID: {log_id} | Reason: {eval_result.get('reason')}")

# --- Interactive Telemetry Inspector ---
st.markdown("---")
st.subheader("Cryptographic Audit & Telemetry Inspector")

if st.session_state.telemetry_logs:
    df_logs = pd.DataFrame(st.session_state.telemetry_logs)
    st.dataframe(df_logs, use_container_width=True)
    
    csv_data = df_logs.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Export Legal Compliance Audit Log (CSV)",
        data=csv_data,
        file_name=f"pdp_audit_log_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/css"
    )
else:
    st.caption("No compliance telemetry recorded yet.")
