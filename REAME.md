
`data/` has no data in it on purpose --- to use the code this needs to be 
populated with a csv file containing the output from `fb_records_to_csv.r`.

To make an entire chatbot, do the following:
 1. Download Facebook data from [https://www.facebook.com/help/212802592074644?helpref=faq_content](https://www.facebook.com/help/212802592074644?helpref=faq_content) as html files.
 2. Run `fb_records_to_csv.r` on the messeges html page. This will generate a csv containing ("input text", "response text") on each line.
 3. Run `chat_bot_preprocessor.py` on the resulting file. This generates the file that stores the Lang object.
 This object stores the word counts and vocabulary for the model.
 4. Run `train_model.py` with the csv file. Note that this involves changing the filename hyperparameter.
 This trains the model and stores the results in `saved_model/`.
 5. Run `chat_bot.py` with the saved model. This will generate a REPL that evaluates the model on a input sentence entered by the user.
