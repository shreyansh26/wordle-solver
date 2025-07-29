LOG_FILE_PATH = "logs/create_sft_data_moonshotai_Kimi-K2-Instruct.log"

word_errors = set()

file_str = open(LOG_FILE_PATH).read()

with open(LOG_FILE_PATH) as f:
    lines = f.readlines()
    for l in lines:
        if "InternalServerError encountered for word " in l:
            word_error = l.split("InternalServerError encountered for word ")[1].split(" ")[0][:-1] # get rid of full stop
            if "Successfully processed word: " + word_error.lower() in file_str or "Successfully processed word: " + word_error.upper() in file_str:
                continue
            word_errors.add(word_error)

word_errors = sorted(list(word_errors))
print(word_errors)
print(len(word_errors))
