import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import time
import random
start = time.time()
import re
import csv

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from docx import Document
# ---------- Use your actual syllabus list here ----------
subject_syllabus =[
  {
    "unit": "UNIT – 1",
    "title": "Conduction and Transient Heat Transfer",
    "topics": [
      "Introduction to heat transfer and different modes",
      "Physical origins and rate equations",
      "Relationship to thermodynamics",
      "Thermal properties of matter",
      "The heat diffusion equation in Cartesian coordinate",
      "Boundary and initial conditions",
      "Special cases, discussion on 3-D conduction in cylindrical and spherical coordinate systems (no derivation)",
      "One dimensional steady state conduction: Plane wall, cylinder and sphere",
      "Thermal contact resistance",
      "Critical thickness of insulation",
      "Conduction with thermal energy generation: plane wall, radial systems",
      "Heat transfer through rectangular fin: Long fin, short fin with insulated tip and convective tip",
      "Fin efficiency and effectiveness",
      "Lumped parameter analysis",
      "Use of Heisler’s charts for transient conduction in slab, long cylinder and sphere"
    ]
  },
  {
    "unit": "UNIT – 2",
    "title": "Convective Heat Transfer",
    "topics": [
      "Velocity and Thermal boundary layers in laminar and turbulent flow conditions",
      "Local and average convection coefficients",
      "Boundary layer equations for laminar flow and its normalized form",
      "Physical interpretations of relevant non-dimensional numbers",
      "External flow: The flat plate in parallel flow - Laminar (Blasius solution) and turbulent flow",
      "Energy equation, Mixed boundary layer conditions",
      "Flat Plates with Constant Heat flux conditions",
      "Cylinder in Cross Flow",
      "Sphere in a flow",
      "Internal flow: Hydrodynamic and thermodynamic Considerations",
      "Mean Velocity and Velocity Profile in Fully Developed Region",
      "Mean Temperature and Newton’s Law of Cooling",
      "Fully Developed Conditions",
      "Laminar Flow in Circular Tubes: Thermal Analysis and Convection Correlations"
    ]
  },
  {
    "unit": "UNIT – 3",
    "title": "Free Convection",
    "topics": [
      "Governing Equations for Laminar Boundary Layers in Free Convection",
      "Empirical Correlations for External Free Convection Flows",
      "Free convection over Vertical plate, Inclined and Horizontal Plates",
      "Free convection around Long Horizontal Cylinder, Enclosures and Spheres"
    ]
  },
  {
    "unit": "UNIT – 4",
    "title": "Radiation Heat Transfer",
    "topics": [
      "Basic concepts of radiation heat transfer",
      "Radiation heat fluxes and Radiation Intensity",
      "Black body radiation: Planck's distribution, Wein’s law, Stefan-Boltzmann law",
      "Kirchhoff’s law and Lambert’s cosine law",
      "Absorption, Reflection, Transmission, and Emission by real surfaces",
      "View factor and Blackbody Radiation Exchange",
      "Radiation Exchange between grey surfaces in an enclosure",
      "Radiation Shields"
    ]
  },
  {
    "unit": "UNIT – 5",
    "title": "Heat Exchangers",
    "topics": [
      "Thermal design of heat exchangers",
      "Overall heat transfer coefficient",
      "Fouling and fouling factor",
      "Temperature profile of heat exchangers",
      "Log Mean Temperature Difference (LMTD): parallel & counter flow",
      "LMTD correction factor",
      "Heat transfer effectiveness",
      "NTU methods of analysis of heat exchangers"
    ]
  }
]

global question_styles
question_styles = [
        "Explain", 
        "Analyze", 
        "Justify", 
        "Design", 
        "Illustrate", 
        "Elucidate", 
        "Interpret", 
        "Infer", 
        "Implement", 
        "Examine", 
        "Sketch", 
        "Distinguish"
    ]

global temp
temp=question_styles
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

checkpoint_dir = "./checkpoints/fine_tuned_mistral/checkpoint-222"
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
    return topics[:n//6], topics[n//6:2*n//6], topics[2*n//6:3*n//6], topics[3*n//6:4*n//6], topics[4*n//6:5*n//6], topics[5*n//6:]

def split_topics_into2(topics):
   n=len(topics)
   return topics[:n//4], topics[n//4:2*n//4], topics[2*n//4:3*n//4], topics[3*n//4:]
# ---------- Question generation ----------
def generate_question(unit, title, topics, marks, numerical=False):
    
    question_type = random.choice(temp)
    temp.remove(question_type)
    print(question_type)
    style_hint = {
        "Explain": "Ask for a clear explanation of a concept.",
        "Analyze": "Pose a question that breaks down a concept or situation into parts and explores relationships.",
        "Justify": "Ask the student to support a decision or method with logical reasoning and evidence.",
        "Design": "Ask for a system layout, structured plan, or model relevant to the topic.",
        "Illustrate": "Ask for a description with suitable examples, diagrams, or scenarios.",
        "Elucidate": "Ask the student to clarify a complex concept in simple terms.",
        "Interpret": "Pose a question that asks for the meaning or implications of a concept or data.",
        "Infer": "Ask the student to draw conclusions based on given information or data.",
        "Implement": "Ask how a theoretical concept can be applied or executed practically.",
        "Examine": "Request a detailed analysis or investigation of a topic or problem.",
        "Sketch": "Ask for a rough drawing or layout to demonstrate a concept visually.",
        "Distinguish": "Ask for comparison or highlighting of differences between two or more concepts."
    }
    if marks == 6:
        effort_str = "6–8 minutes and about 100 words"
    elif marks == 10:
        effort_str = "15 minutes and about 200 words"
    else:
        effort_str = f"{int(marks * 1.5)} minutes and about {marks * 20} words"
    toadd=style_hint[question_type]
    if numerical:
       effort_str += ".\n Make this question a numerical problem on the above topic."
       toadd=" \"Numerical\": \"Give a numerical, create your own question based on the topic.\" "
    prompt = (
        f"Generate one descriptive question from UNIT {unit}: {title}.\n"
        f"Topics: {', '.join(topics)}.\n"
        f"Choose any one topic. The question should be worth {marks} marks.\n"
        f"The question should be concise and appropriate for a {marks}-mark answer.\n"
        f"The expected answer should take around {effort_str}.\n"
        f"No MCQs. Avoid questions that would require a prerequisite image. Avoid overly broad or multi-part questions.\n"
        f"{toadd}\n"
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
        global temp
        temp=[
            "Explain", 
            "Analyze", 
            "Justify", 
            "Design", 
            "Illustrate", 
            "Elucidate", 
            "Interpret", 
            "Infer", 
            "Implement", 
            "Examine", 
            "Sketch", 
            "Distinguish"
        ]
        if options[i] == 1:
          part1, part2, part3, part4, part5, part6 = split_topics(topics)
          q1 = generate_question(unit, title, part1, 6)
          q2 = generate_question(unit, title, part3, 6)
          q3 = generate_question(unit, title, part5, 8)
          question_paper += f"\n<unit> {unit} - {marks_per_unit} Marks: {title} </unit>\n FQ1 \n"
          question_paper += f"<qno> {2*i+1} a) </qno> {q1} <marks>(6 marks)</marks>\n "
          question_paper += f"<qno> {2*i+1}b) </qno> {q2} <marks>(6 marks)</marks>\n "
          question_paper += f"<qno> {2*i+1}c) </qno> {q3} <marks>(8 marks)</marks>\n\n"


          q1 = generate_question(unit, title, part2, 6)
          q2 = generate_question(unit, title, part4, 6)
          q3 = generate_question(unit, title, part6, 8)
          question_paper += f"\n<unit> {unit} - {marks_per_unit} Marks: {title} </unit>\n FQ2 \n"
          question_paper += f"<qno> {2*i + 2} a) </qno> {q1} <marks>(6 marks)</marks>\n "
          question_paper += f"<qno> {2*i+2}b) </qno> {q2} <marks>(6 marks)</marks>\n "
          question_paper += f"<qno> {2*i+2}c) </qno> {q3} <marks>(8 marks)</marks>\n\n"
        
        elif options[i] == 2:
          part1, part2, part3, part4, part5, part6 = split_topics(topics)
          q1 = generate_question(unit, title, part1, 4)
          q2 = generate_question(unit, title, part5, 8)
          q3 = generate_question(unit, title, part3, 8)
          question_paper += f"\n <unit> {unit} - {marks_per_unit} Marks: {title} </unit>\n FQ1 \n"
          question_paper += f"<qno> {2*i+1}a)</qno> {q1} <marks>(4 marks)</marks>\n "
          question_paper += f"<qno>{2*i+1}b)</qno> {q2} <marks>(8 marks)</marks>\n "
          question_paper += f"<qno>{2*i+1}c)</qno> {q3} <marks>(8 marks)</marks>\n\n"

          q1 = generate_question(unit, title, part2, 4)
          q2 = generate_question(unit, title, part4, 8)
          q3 = generate_question(unit, title, part6, 8)
          question_paper += f"\n <unit> {unit} - {marks_per_unit} Marks: {title} </unit>\n FQ2 \n"
          question_paper += f"<qno>{2*i+2}a)</qno> {q1} <marks>(4 marks)</marks>\n "
          question_paper += f"<qno>{2*i+2}b)</qno> {q2} <marks>(8 marks)</marks>\n "
          question_paper += f"<qno>{2*i+2}c)</qno> {q3} <marks>(8 marks)</marks>\n\n"
        


        elif options[i] == 3:
          part1, part2, part3, part4 = split_topics_into2(topics)
          q1 = generate_question(unit, title, part1, 10)
          q2 = generate_question(unit, title, part3, 10)
          question_paper += f"\n <unit> {unit} - {marks_per_unit} Marks: {title} </unit>\n FQ1 \n"
          question_paper += f"<qno>{2*i+1}a)</qno> {q1} <marks>(10 marks)</marks>\n "
          question_paper += f"<qno>{2*i+1}b)</qno> {q2} <marks>(10 marks)</marks>\n "

          q1 = generate_question(unit, title, part2, 10)
          q2 = generate_question(unit, title, part4, 10)
          question_paper += f"\n <unit> {unit} - {marks_per_unit} Marks: {title} </unit>\n FQ2 \n"
          question_paper += f"<qno>{2*i+2}a)</qno> {q1} <marks>(10 marks)</marks>\n "
          question_paper += f"<qno>{2*i+2}b)</qno> {q2} <marks>(10 marks)</marks>\n "

        elif options[i]==4:
          part1, part2, part3, part4, part5, part6 = split_topics(topics)
          q1 = generate_question(unit, title, part1, 5)
          q2 = generate_question(unit, title, part3, 5)
          q3 = generate_question(unit, title, part5, 10)
          question_paper += f"\n <unit> {unit} - {marks_per_unit} Marks: {title} </unit> \n FQ1 \n"
          question_paper += f"<qno>{2*i+1}a)</qno> {q1} <marks>(5 marks)</marks>\n "
          question_paper += f"<qno>{2*i+1}b)</qno> {q2} <marks>(5 marks)</marks>\n "
          question_paper += f"<qno>{2*i+1}c)</qno> {q3} <marks>(10 marks)</marks>\n\n"

          q1 = generate_question(unit, title, part2, 5)
          q2 = generate_question(unit, title, part4, 5)
          q3 = generate_question(unit, title, part6, 10)
          question_paper += f"\n <unit> {unit} - {marks_per_unit} Marks: {title} </unit> \n FQ2 \n"
          question_paper += f"<qno>{2*i+2}a)</qno> {q1} <marks>(5 marks)</marks>\n "
          question_paper += f"<qno>{2*i+2}b)</qno> {q2} <marks>(5 marks)</marks>\n "
          question_paper += f"<qno>{2*i+2}c)</qno> {q3} <marks>(10 marks)</marks>\n\n"


        else:
          part1, part2, part3, part4 = split_topics_into2(topics)
          q1 = generate_question(unit, title, part1, 10)
          q2 = generate_question(unit, title, part3, 10, numerical=True)
          question_paper += f"\n <unit> {unit} - {marks_per_unit} Marks: {title} </unit>\n FQ1 \n"
          question_paper += f"<qno>{2*i+1}a)</qno> {q1} <marks>(10 marks)</marks>\n "
          question_paper += f"<qno>{2*i+1}b)</qno> {q2} <marks>(10 marks)</marks>\n "
          
          q1 = generate_question(unit, title, part2, 10)
          q2 = generate_question(unit, title, part4, 10, numerical=True)
          question_paper += f"\n <unit> {unit} - {marks_per_unit} Marks: {title} </unit> FQ1 \n"
          question_paper += f"<qno>{2*i+2}a)</qno> {q1} <marks>(10 marks)</marks>\n "
          question_paper += f"<qno>{2*i+2}b)</qno> {q2} <marks>(10 marks)</marks>\n "
        
        i+=1

    return question_paper

# ---------- Run ----------
question_paper = generate_question_paper(subject_syllabus)



# After question_paper is generated



# Write the cleaned output to a .txt file
print(question_paper)

# Save to file
with open("generated_question_paper_dbms.txt", "w") as f:
    f.write(question_paper)

genai.configure(api_key='AIzaSyAlWRDEGQZgNUTu_4dL1HwxE0Fwp1I8CqU')

model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash-preview-04-17",  # or use "gemini-2.5-pro" if accessible
    generation_config=GenerationConfig(
        temperature=0.3,
        max_output_tokens=20000,
    )
)

prompt2 = f"""
Please clean and format the following raw question paper text. The output should be organized like a csv as follows:

unit_name, question_no, question_text, marks

The question paper has 5 units, 2 full questions from each unit.
Ensure:
- No XML or HTML tags (like <question> or </question>)
- Proper structure, indentation, and line breaks
- Ready to be copied into a csv file
- No other extra response text, just give me the question paper text in csv format.
- Be careful, some questions have commas, which causes confusion. Put each question in quotes and return the text so that it is ready to save as csv file.

Text:
{question_paper}
"""

response = model.generate_content(prompt2)
print("This is the response", response)
cleaned_text = response.text

with open('Heat.csv', 'w', newline='', encoding='utf-8') as f:
    f.write(cleaned_text)

print("CSV file saved as 'heat_qp.csv'")


print(f"\nExecution time: {round(time.time() - start, 2)} seconds")
