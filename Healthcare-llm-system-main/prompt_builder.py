# ==============================================================================
# FILE: PROMPT_BUILDER.PY
# PURPOSE: Dynamic Prompt Builder for LLM Integration
# ==============================================================================

from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PromptContext:
    """Context information for building prompts"""
    user_input: str
    predicted_department: str
    classification_confidence: float
    retrieved_document: Dict[str, Any]
    severity_analysis: Dict[str, Any]

class PromptBuilder:
    """Dynamic prompt builder for medical symptom analysis"""
    
    def __init__(self):
        self.system_prompt_template = """You are a helpful medical assistant. Your role is to:
1. Analyze the patient's symptoms based on the provided medical information
2. Explain the potential causes and implications in simple, understandable terms
3. Provide clear, actionable recommendations
4. Always emphasize when immediate medical attention is needed
5. Be empathetic and professional in your response

IMPORTANT: This is for informational purposes only and should not replace professional medical advice."""

    def build_medical_prompt(self, context: PromptContext) -> str:
        """Build a comprehensive medical analysis prompt"""
        
        doc = context.retrieved_document
        
        # Build the main prompt
        prompt = f"""{self.system_prompt_template}

PATIENT SYMPTOMS:
{context.user_input}

MEDICAL ANALYSIS CONTEXT:
Department: {context.predicted_department} (confidence: {context.classification_confidence:.1%})
Symptom Match: {doc['symptom']}
Urgency Level: {context.severity_analysis['urgency_level']}
Severity Assessment: {context.severity_analysis['severity']}

RELEVANT MEDICAL INFORMATION:
Symptoms: {', '.join(doc.get('aliases', []))}
Potential Causes: {', '.join(doc.get('causes', []))}
Possible Consequences: {', '.join(doc.get('consequences', []))}
Medical Recommendations: {', '.join(doc.get('suggestions', []))}

TASK:
Based on the above information, provide a comprehensive, natural, and empathetic analysis that includes:

1. A clear, conversational explanation of what might be happening with the patient's symptoms
2. The potential seriousness of the situation in simple terms
3. Specific, actionable recommendations for the patient
4. When to seek immediate medical attention
5. What the patient can expect

IMPORTANT: Write in a natural, conversational tone as if you're speaking directly to the patient. Be empathetic, clear, and informative. Avoid generic warnings - provide specific, helpful information based on the medical context provided.

Example format:
"It sounds like you're experiencing [symptoms]. These could be signs that [explanation]. This can happen due to things like [causes]. It's important to [why it matters]. I recommend [specific actions]. Also, try to [additional advice]."

Please respond in a friendly, empathetic tone while being medically accurate and clear about urgency levels."""

        return prompt
    
    def build_emergency_prompt(self, context: PromptContext) -> str:
        """Build a specialized prompt for emergency situations"""
        
        doc = context.retrieved_document
        
        prompt = f"""{self.system_prompt_template}

🚨 EMERGENCY MEDICAL SITUATION 🚨

PATIENT SYMPTOMS:
{context.user_input}

CRITICAL INFORMATION:
Department: {context.predicted_department}
Symptom Match: {doc['symptom']}
Urgency Level: {context.severity_analysis['urgency_level']} - REQUIRES IMMEDIATE ATTENTION

EMERGENCY MEDICAL CONTEXT:
Symptoms: {', '.join(doc.get('aliases', []))}
Potential Causes: {', '.join(doc.get('causes', []))}
Critical Consequences: {', '.join(doc.get('consequences', []))}
Emergency Actions: {', '.join(doc.get('suggestions', []))}

TASK:
This appears to be a potentially serious medical situation. Provide:
1. A clear, urgent explanation of the situation
2. Immediate actions the patient should take
3. Strong emphasis on seeking emergency medical care
4. What to expect and why immediate attention is crucial

Use urgent but calm language. Emphasize the importance of immediate medical attention."""

        return prompt
    
    def build_general_prompt(self, context: PromptContext) -> str:
        """Build a general medical advice prompt"""
        
        doc = context.retrieved_document
        
        prompt = f"""{self.system_prompt_template}

PATIENT SYMPTOMS:
{context.user_input}

MEDICAL CONTEXT:
Department: {context.predicted_department}
Symptom Match: {doc['symptom']}
Urgency Level: {context.severity_analysis['urgency_level']}

RELEVANT INFORMATION:
Symptoms: {', '.join(doc.get('aliases', []))}
Possible Causes: {', '.join(doc.get('causes', []))}
General Recommendations: {', '.join(doc.get('suggestions', []))}

TASK:
Provide helpful medical guidance that includes:
1. A clear explanation of the symptoms
2. Common causes and what they mean
3. Self-care recommendations
4. When to consult a healthcare provider
5. What to monitor for worsening symptoms

Be reassuring while encouraging appropriate medical follow-up when needed."""

        return prompt
    
    def build_prompt(self, context: PromptContext) -> str:
        """Main method to build appropriate prompt based on urgency"""
        
        urgency_level = context.severity_analysis['urgency_level']
        
        if urgency_level in ['emergency', 'high']:
            return self.build_emergency_prompt(context)
        else:
            return self.build_general_prompt(context)
    
    def build_followup_prompt(self, original_context: PromptContext, user_followup: str) -> str:
        """Build a follow-up prompt for additional questions"""
        
        doc = original_context.retrieved_document['document']
        
        prompt = f"""{self.system_prompt_template}

FOLLOW-UP QUESTION:
{user_followup}

ORIGINAL CONTEXT:
Patient's original symptoms: {original_context.user_input}
Department: {original_context.predicted_department}
Symptom Match: {doc.title}
Urgency Level: {original_context.severity_analysis['urgency_level']}

RELEVANT MEDICAL INFORMATION:
Symptoms: {', '.join(doc.symptoms)}
Causes: {', '.join(doc.causes)}
Recommendations: {', '.join(doc.recommendations)}

TASK:
Answer the patient's follow-up question while:
1. Maintaining consistency with the original medical context
2. Providing additional relevant information
3. Addressing any new concerns raised
4. Reinforcing appropriate medical guidance

Be helpful and informative while staying within the scope of the original medical context."""

        return prompt
    
    def build_summary_prompt(self, context: PromptContext) -> str:
        """Build a summary prompt for quick overview"""
        
        doc = context.retrieved_document['document']
        
        prompt = f"""{self.system_prompt_template}

PATIENT SYMPTOMS:
{context.user_input}

QUICK SUMMARY REQUESTED:
Department: {context.predicted_department}
Symptom Match: {doc.title}
Urgency: {context.severity_analysis['urgency_level']}

KEY POINTS:
- Symptoms: {', '.join(doc.symptoms)}
- Main causes: {', '.join(doc.causes[:2])}  # Top 2 causes
- Key recommendations: {', '.join(doc.recommendations[:2])}  # Top 2 recommendations

TASK:
Provide a concise, 2-3 sentence summary that includes:
1. What the symptoms likely indicate
2. The urgency level
3. The most important next step

Keep it brief but informative."""

        return prompt

def create_prompt_builder() -> PromptBuilder:
    """Create a prompt builder instance"""
    return PromptBuilder()

if __name__ == "__main__":
    # Test the prompt builder
    builder = create_prompt_builder()
    
    # Mock context for testing
    from knowledge_base import SymptomDocument
    
    mock_doc = SymptomDocument(
        id="test_001",
        title="Chest Pain and Shortness of Breath",
        symptoms=["chest pain", "shortness of breath"],
        department="cardiology",
        causes=["angina", "heart attack"],
        consequences=["heart damage"],
        recommendations=["seek immediate medical attention"],
        urgency_level="emergency",
        content="Test content"
    )
    
    mock_context = PromptContext(
        user_input="I have chest pain and can't breathe",
        predicted_department="cardiology",
        classification_confidence=0.85,
        retrieved_document={
            'document': mock_doc,
            'enhanced_score': 0.92
        },
        severity_analysis={
            'severity': 'critical',
            'urgency_level': 'emergency',
            'recommendations': ['seek immediate medical attention']
        }
    )
    
    # Test different prompt types
    print("🔧 Testing Prompt Builder:")
    
    print("\n1. Emergency Prompt:")
    emergency_prompt = builder.build_emergency_prompt(mock_context)
    print(emergency_prompt[:500] + "...")
    
    print("\n2. General Prompt:")
    general_prompt = builder.build_general_prompt(mock_context)
    print(general_prompt[:500] + "...")
    
    print("\n3. Follow-up Prompt:")
    followup_prompt = builder.build_followup_prompt(mock_context, "How long should I wait before going to the hospital?")
    print(followup_prompt[:500] + "...")
    
    print("\n4. Summary Prompt:")
    summary_prompt = builder.build_summary_prompt(mock_context)
    print(summary_prompt[:500] + "...") 