from transformers import pipeline


def main():
    #Load a gpt-2 text generation pipeline
    generator = pipeline("text-generation", model ="gpt2")

    #Take input from user
    prompt = input("Enter your prompt: ")

    #generate text
    output = generator(
        prompt,
        max_length=100,
        num_return_sequences=1,
        temperature = 0.7,
        top_p = 0.9,
        do_sample=True
    )

    #Display result
    print("\nGenerated text:")
    print(output[0]['generated_text'])

# generator = pipeline("text-generation", model="gpt2")
#
# def generate_text(prompt):
#     output = generator(prompt, max_length=100, temperature=0.7, top_p=0.9, do_sample=True)
#     return output[0]['generated_text']
# gr.Interface(fn=generate_text, inputs="text", outputs="text").launch() #change the name of function to generate_text, remove main call and import gradio as gr
if __name__ =="__main__":
    main()