LOG_FILE_PATH = "logs/create_sft_data_moonshotai_Kimi-K2-Instruct.log"

word_errors = []

with open(LOG_FILE_PATH) as f:
    lines = f.readlines()
    for l in lines:
        if "InternalServerError encountered for word " in l:
            word_error = l.split("InternalServerError encountered for word ")[1].split(" ")[0][:-1] # get rid of full stop
            word_errors.append(word_error)


print(word_errors)