import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import time

start = time.time()

# ---------- Use your actual syllabus list here ----------
subject_syllabus =  [
  {
    "unit": "UNIT – 1",
    "hours": 8,
    "title": "The Machine Learning Landscape and Bayesian Decision Theory",
    "topics": [
      "What Is Machine Learning (ML)?",
      "Uses and Applications with examples",
      "Types of Machine Learning",
      "Main Challenges of Machine Learning",
      "Testing and Validating",
      "End to End Machine Learning",
      "Working with Real Data",
      "Frame the Problem",
      "Select the Performance Measure",
      "Prepare the Data for ML Algorithms",
      "Training and Evaluating the Data Set",
      "Bayesian Decision Theory: Introduction",
      "Classification",
      "Losses and Risks",
      "Discriminant Functions",
      "Association Rules"
    ]
  },
  {
    "unit": "UNIT – 2",
    "hours": 7,
    "title": "Classification and Training Models",
    "topics": [
      "MNIST",
      "Training Binary Classifier",
      "Performance Measures",
      "Multiclass classification",
      "Error Analysis",
      "Multilabel & Multioutput Classifications",
      "Linear Regression",
      "Gradient Descent",
      "Regularized Linear Models – Ridge & Lasso Regression",
      "Logistic Regression"
    ]
  },
  {
    "unit": "UNIT – 3",
    "hours": 7,
    "title": "Dimensionality Reduction and Support Vector Machines",
    "topics": [
      "The Curse of Dimensionality",
      "Main Approaches for Dimensionality",
      "PCA",
      "Kernel PCA",
      "LLE",
      "Linear Discriminant Analysis (LDA)",
      "Linear SVM Classification",
      "Nonlinear SVM",
      "SVM Regression",
      "Kernelized SVMs"
    ]
  },
  {
    "unit": "UNIT – 4",
    "hours": 7,
    "title": "Decision Trees",
    "topics": [
      "Univariate Trees: classification & Regression Trees",
      "Training and Visualizing a Decision Tree",
      "Pruning",
      "Rule Extraction from Trees",
      "Learning Rules from Data",
      "Making Predictions",
      "Estimating Class Probabilities",
      "CART Training Algorithm",
      "Computational Complexity",
      "Gini Impurity or Entropy?",
      "Regularization Hyperparameters",
      "Multivariate Trees"
    ]
  },
  {
    "unit": "UNIT – 5",
    "hours": 7,
    "title": "Ensemble Learning and Unsupervised Learning Techniques",
    "topics": [
      "Voting Classifiers",
      "Bagging and Pasting",
      "Random Patches and Random Subspaces",
      "Random Forests",
      "Boosting",
      "Clustering – K means",
      "Spectral Clustering",
      "Hierarchical Clustering"
    ]
  }
]



options=[]
print("There are 3 options for makrs split u of a unit!! \n 1) 6+6+8 \n 2)4+8+8 \n 3) 10+10 \n 4)5+5+10\n 5)10+10 with numerical \n Choose any 1.")
for i in range(1,6):
    option = int(input(f"Enter marks split up choice for unit {i}: "))
    options.append(option)
print(options)
# ---------- Load tokenizer and model ----------
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

checkpoint_dir = "./checkpoints/fine_tuned_mistral/checkpoint-90"
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Apply LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)

# ---------- Function to split topics ----------
def split_topics(topics):
    n = len(topics)
    return topics[:n//3], topics[n//3:2*n//3], topics[2*n//3:]

def split_topics_into2(topics):
   n=len(topics)
   return topics[:n//2], topics[n//2:]
# ---------- Question generation ----------
def generate_question(unit, title, topics, marks, numerical=False):
    if marks == 6:
        effort_str = "6–8 minutes and about 100 words"
    elif marks == 10:
        effort_str = "15 minutes and about 200 words"
    else:
        effort_str = f"{int(marks * 1.5)} minutes and about {marks * 20} words"
    if numerical:
       effort_str += ".\n Make this question a numerical problem on the above topic."
    prompt = (
        f"Generate one descriptive question from UNIT {unit}: {title}.\n"
        f"Topics: {', '.join(topics)}.\n"
        f"Choose any one topic. The question should be worth {marks} marks.\n"
        f"The question should be concise and appropriate for a {marks}-mark answer.\n"
        f"The expected answer should take around {effort_str}.\n"
        f"No MCQs. Avoid questions that would require a prerequisite image. Avoid overly broad or multi-part questions."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones(inputs['input_ids'].shape, device=model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_length=400,
            num_beams=5,
            no_repeat_ngram_size=2,
            temperature=0.69,
            top_p=0.9,
            top_k=50,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ---------- Full question paper ----------
def generate_question_paper(syllabus):
    question_paper = ""
    marks_per_unit = 20
    i=0
    for unit_data in syllabus:
        print(options[i])
        unit = unit_data["unit"]
        title = unit_data["title"]
        topics = unit_data["topics"]
        if options[i] == 1:
          part1, part2, part3 = split_topics(topics)
          q1 = generate_question(unit, title, part1, 6)
          q2 = generate_question(unit, title, part2, 6)
          q3 = generate_question(unit, title, part3, 8)
          question_paper += f"\n {unit} - {marks_per_unit} Marks: {title} \n"
          question_paper += f"a) {q1} (6 marks)\n "
          question_paper += f"b) {q2} (6 marks)\n "
          question_paper += f"c) {q3} (8 marks)\n\n"
        
        elif options[i] == 2:
          part1, part2, part3 = split_topics(topics)
          q1 = generate_question(unit, title, part1, 4)
          q2 = generate_question(unit, title, part2, 8)
          q3 = generate_question(unit, title, part3, 8)
          question_paper += f"\n {unit} - {marks_per_unit} Marks: {title} \n"
          question_paper += f"a) {q1} (4 marks)\n "
          question_paper += f"b) {q2} (8 marks)\n "
          question_paper += f"c) {q3} (8 marks)\n\n"
        

        elif options[i] == 3:
          part1, part2 = split_topics_into2(topics)
          q1 = generate_question(unit, title, part1, 10)
          q2 = generate_question(unit, title, part2, 10)
          question_paper += f"\n {unit} - {marks_per_unit} Marks: {title} \n"
          question_paper += f"a) {q1} (10 marks)\n "
          question_paper += f"b) {q2} (10 marks)\n "

        elif options[i]==4:
          part1, part2, part3 = split_topics(topics)
          q1 = generate_question(unit, title, part1, 5)
          q2 = generate_question(unit, title, part2, 5)
          q3 = generate_question(unit, title, part3, 10)
          question_paper += f"\n {unit} - {marks_per_unit} Marks: {title} \n"
          question_paper += f"a) {q1} (5 marks)\n "
          question_paper += f"b) {q2} (5 marks)\n "
          question_paper += f"c) {q3} (10 marks)\n\n"

        else:
          part1, part2 = split_topics_into2(topics)
          q1 = generate_question(unit, title, part1, 10, numerical=True)
          q2 = generate_question(unit, title, part2, 10)
          question_paper += f"\n {unit} - {marks_per_unit} Marks: {title} \n"
          question_paper += f"a) {q1} (10 marks)\n "
          question_paper += f"b) {q2} (10 marks)\n "
           
        
        i+=1

    return question_paper

# ---------- Run ----------
question_paper = generate_question_paper(subject_syllabus)

import re

# After question_paper is generated

# Extract all text between  and 
matches = re.findall(r"(.*?)", question_paper, re.DOTALL)

# Join matches with newlines and preserve original formatting
output_text = "\n".join(match.strip() for match in matches)

# Write the cleaned output to a .txt file
print(question_paper)

# Save to file
with open("generated_question_paper_iml2.txt", "w") as f:
    f.write(question_paper)

print(f"\nExecution time: {round(time.time() - start, 2)} seconds")
