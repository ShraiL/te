# ═══════════════════════════════════════════════════════════════
# BLIP_CAPTION.PY - Image to Text
# ═══════════════════════════════════════════════════════════════

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import BlipForQuestionAnswering
from PIL import Image

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CHANGE THESE                                                 ║
# ╚═══════════════════════════════════════════════════════════════╝
IMAGE_PATH = r"images/test.jpg"      # Path to your image
TEXT_PROMPT = "a photo of"           # For conditional caption
QUESTION = "What is in the image?"   # For Q&A


# Load models
print("Loading models...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
qa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
print("Models loaded!")

# Load image
image = Image.open(IMAGE_PATH)
print(f"Image loaded: {IMAGE_PATH}")


# ═══════════════════════════════════════════════════════════════
# METHOD 1: Basic Caption (Unconditional)
# ═══════════════════════════════════════════════════════════════
def basic_caption(img):
    """Generate caption without any prompt"""
    inputs = processor(img, return_tensors="pt")
    outputs = caption_model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption


# ═══════════════════════════════════════════════════════════════
# METHOD 2: Conditional Caption
# ═══════════════════════════════════════════════════════════════
def conditional_caption(img, text_prompt):
    """Generate caption starting with the given prompt"""
    inputs = processor(img, text_prompt, return_tensors="pt")
    outputs = caption_model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption


# ═══════════════════════════════════════════════════════════════
# METHOD 3: Visual Question Answering
# ═══════════════════════════════════════════════════════════════
def visual_qa(img, question):
    """Answer a question about the image"""
    inputs = processor(img, question, return_tensors="pt")
    outputs = qa_model.generate(**inputs)
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer


if __name__ == '__main__':
    print("\n" + "=" * 50)
    
    # Basic caption
    caption1 = basic_caption(image)
    print(f"Basic Caption: {caption1}")
    
    # Conditional caption
    caption2 = conditional_caption(image, TEXT_PROMPT)
    print(f"Conditional Caption: {caption2}")
    
    # Question answering
    answer = visual_qa(image, QUESTION)
    print(f"Question: {QUESTION}")
    print(f"Answer: {answer}")
    
    print("=" * 50)