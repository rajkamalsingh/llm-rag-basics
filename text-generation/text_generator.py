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

if __name__ =="__main__":
    main()