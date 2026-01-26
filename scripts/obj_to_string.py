import sys
import json
import os

def main():
    # 1. Check arguments
    if len(sys.argv) < 2:
        print("Error: Please provide a filename.")
        print("Usage: python file_to_json.py <path_to_file>")
        sys.exit(1)

    input_filename = sys.argv[1]

    # 2. Define the output filename (e.g., base_0.obj -> base_0.obj.txt)
    output_filename = input_filename + ".txt"

    try:
        # 3. Read the input file
        with open(input_filename, 'r', encoding='utf-8') as f_in:
            content = f_in.read()
        
        # 4. Convert to JSON string
        json_string = json.dumps(content)
        
        # 5. Write to the output file
        with open(output_filename, 'w', encoding='utf-8') as f_out:
            f_out.write(json_string)

        print(f"✅ Success! Encoded string saved to: {output_filename}")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()