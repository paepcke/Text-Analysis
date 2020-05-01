#  This code will read an input csv file and break specified text fields into words and n-grams.
#  Basic sentiment scores will also be assigned to each word and n-gram.
#  Stop words will also be identified in the word breakout.
#  Two csv files (one for words and one for n-grams) will be written.

#  To add more stop words, edit the language file in nltk_data\corpora\stopwords

from collections import defaultdict
import csv
from functools import lru_cache
import math
import os
import re
import sys
import argparse

import nltk
from nltk.corpus import stopwords 
from nltk.sentiment import sentiment_analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.translate.gdfa import grow_diag_final_and
from nltk.util import ngrams
from pip._vendor.requests.api import head


class TextAnalyzer(object):

    stopWordLanguageList = ['arabic','azerbaijani','danish','dutch',
                            'english','finnish','french','german','greek',
                            'hungarian','indonesian','italian','kazakh',
                            'nepali','norwegian','portuguese','romanian',
                            'russian','slovene','spanish','swedish','tajik','turkish'
                            ]
    stopword_language = 'english'
    
    word_file_suffix = '_wordstats'
    ngram_file_suffix = '_ngrams'

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, 
                 text_file,
                 text_fields,
                 outfiles_dir=None,
                 record_id_col='id',
                 stopword_language='english',
                 ngram_len=3,
                 ):
        
        if outfiles_dir is not None:
            if not os.path.exists(outfiles_dir):
                os.mkdir(outfiles_dir)
            elif not os.path.isdir(outfiles_dir):
                print(f"Outfile directory is a file: {outfiles_dir}; quitting")
                sys.exit(1)

        self.text_file = text_file
        self.outfiles_dir = outfiles_dir
        self.text_fields = text_fields
        self.record_id_col = record_id_col
        self.stopword_language = stopword_language
        self.ngram_len = ngram_len

        self.create_wordnet_nltk_pos_xlation()
        # Generate outfile names for word stats,
        # and ngrams:

        if outfiles_dir is None:
            # Output result files to same dir as infile:
            (infile_path, ext) = os.path.splitext(text_file)
            wordstats_path = f"{infile_path}{self.word_file_suffix}.csv"
            ngrams_path = f"{infile_path}{self.ngram_file_suffix}.csv"
        else:
            # Outdir different from infile's:
            wordstats_path = os.path.join(outfiles_dir,
                                          f"{os.path.basename(text_file)}{self.word_file_suffix}.csv")

        self.generate_outfiles(text_file,
                               wordstats_path,
                               ngrams_path)

    #------------------------------------
    # generate_outfiles
    #-------------------
    
    def generate_outfiles(self, 
                          text_file,
                          word_stats_path,
                          ngrams_path):
        '''
        Assumptions about inst var initializations:
            o text_fields,
            o record_id_col,
            o stopword_language,
            o ngram_len
            o pos_tag_map
             
        @param text_file:
        @type text_file:
        @param outfiles_dir:
        @type outfiles_dir:
        '''

        try:
            # Open outfiles:
            word_stats_fd = open(word_stats_path, 'w', newline='')
            ngram_fd      = open(ngrams_path, 'w', newline='')
            
            # The words stats .csv writer
            word_stats_cols = [self.record_id_col,
                               'word',
                               'stem',
                               'lemmatized',
                               'pos',
                               'sent_neg',
                               'sent_neu',
                               'sent_pos',
                               'sent_compound',
                               'stop_word',
                               'word_number']
            word_stats_writer = csv.DictWriter(word_stats_fd, word_stats_cols)
            word_stats_writer.writeheader()

            # The ngrams .csv writer. The col header depends
            # on the number of ngrams requested:
            #   id,Word1,Word2,...,full_ngram, ngram_sentiment, ngram_number
            
            self.word_col_names =[f"Word{indx+1}" for indx in range(self.ngram_len)]
            heading = [self.record_id_col]
            heading.extend(self.word_col_names)

                
            heading.extend(['full_ngram', 'ngram_sentiment', 'ngram_number'])
            ngrams_writer     = csv.DictWriter(ngram_fd, heading)
            ngrams_writer.writeheader()
            
            # The NLTK tools:
            tokenizer  = TweetTokenizer(preserve_case=False)
            stemmer    = SnowballStemmer(self.stopword_language)
            # Speed up stemming:
            self.stem = lru_cache(maxsize=50000)(stemmer.stem)
            lemmatizer = WordNetLemmatizer()
            # Speed up lemmatization by caching:
            self.lemmatize = lru_cache(maxsize=50000)(lemmatizer.lemmatize)
            # Create the sentiment analyzer for some basic sentiment tests.
            sentiment_analyzer = SentimentIntensityAnalyzer()

            # List of all_stopwords:
            self.all_stopwords = set(stopwords.words(self.stopword_language)) 
            
            # Regex to keep only alpha and apostrophe.
            # Eliminates punctuation, but keeps contractions,
            # such as "can't":
            spec_char_pat = re.compile(r"[\w']+")

            record_num = 0
            with open(text_file, 'r') as in_fd:
                csv_reader = csv.DictReader(in_fd)
                for row_dict in csv_reader:
                    record_num += 1
                    for txt_field in self.text_fields:
                        text = row_dict[txt_field]
                        # Tokenize:
                        token_arr = tokenizer.tokenize(text)
                        # Remove punctuation:
                        clean_token_arr = [kept_token for kept_token \
                                           in token_arr \
                                           if spec_char_pat.match(kept_token) \
                                           is not None]
                        self.write_ngrams(record_num, 
                                          clean_token_arr, 
                                          ngrams_writer, 
                                          sentiment_analyzer, 
                                          self.ngram_len)
                        self.write_word_stats(record_num,
                                              row_dict,
                                              text,
                                              clean_token_arr, 
                                              word_stats_writer, 
                                              sentiment_analyzer)

        finally:
            word_stats_fd.close()
            ngram_fd.close()
                
    #------------------------------------
    # write_word_stats 
    #-------------------
    
    def write_word_stats(self,
                         record_num,
                         row_dict,
                         text,
                         clean_token_arr, 
                         word_stats_writer, 
                         sentiment_analyzer):
                
        # First, get sentiment neg/neu/pos/compount for
        # the text:
        sent_dict = sentiment_analyzer.polarity_scores(text)
        sent_arr  = [sent_dict['neg'],
                     sent_dict['neu'],
                     sent_dict['pos'],
                     sent_dict['compound'],
                     ]

        stem_arr  = [self.stem(word) for word in clean_token_arr]
        pos_tuples = nltk.pos_tag(clean_token_arr)
        lem_arr   = [self.lemmatize(word, pos=self.pos_tag_map[pos]) \
                     for (word,pos) in pos_tuples]
        pos_arr   = [pos for (word,pos) in pos_tuples]
        stopword_status_arr = [word in self.all_stopwords for word in clean_token_arr]
        
        # Finally, put the rows together, a row for each value:
        out_row = {}
        for (i, token) in enumerate(clean_token_arr):
            out_row[self.record_id_col] = row_dict[self.record_id_col]
            out_row['word'] = token
            out_row['stem'] = stem_arr[i]
            out_row['lemmatized'] = lem_arr[i]
            out_row['pos'] = pos_arr[i]
            out_row['sent_neg'] = sent_dict['neg'] 
            out_row['sent_neu'] = sent_dict['neu']
            out_row['sent_pos'] = sent_dict['pos']
            out_row['sent_compound'] = sent_dict['compound']
            out_row['stop_word'] = stopword_status_arr[i]
            out_row['word_number'] = record_num
            
        word_stats_writer.writerow(out_row)

    #------------------------------------
    # write_ngrams
    #-------------------

    
    def write_ngrams(self, row_id, clean_token_arr, ngram_writer, nltkSentiment, ngram_len):
        
        ngram_tuples = list(ngrams(clean_token_arr, ngram_len))
        
        for (i, ngram_tuple) in enumerate(ngram_tuples):
            out_dict = dict(list(zip(self.word_col_names, ngram_tuple)))
            full_ngram = ' '.join(ngram_tuple)
            out_dict['full_ngram'] = full_ngram
            out_dict['ngram_sentiment'] = nltkSentiment.polarity_scores(full_ngram)['compound']
            out_dict['ngram_number'] = str(i)
            out_dict[self.record_id_col] = row_id
            ngram_writer.writerow(out_dict)
        
    
    #------------------------------------
    # tokenize
    #-------------------

    def tokenize(self, text_file):
        pass

    #------------------------------------
    # create_wordnet_nltk_pos_xlation
    #-------------------
    
    def create_wordnet_nltk_pos_xlation(self):
        self.pos_tag_map = defaultdict(lambda : 'n')
        self.pos_tag_map['J'] = 'a'
        self.pos_tag_map['V'] = 'v'
        self.pos_tag_map['R'] = 'r'

    #*******************

#     # Valid stop word language?
#     if not(stopword_language in stopWordLanguageList):
#          sys.exit("Invalid stop word language. Exiting program.")   
#     
#     # Check to make sure the input file exists:
#     if not(os.path.exists(text_file)):
#         sys.exit("Input file does not exits. Exiting the program.")
#     
#     # Delete any previously written files.
#     outFile = os.path.join(outfiles_dir, "words.csv")
#     if os.path.exists(outFile):
#         os.remove(outFile) 
#     
#     outFile = os.path.join(outfiles_dir, "ngrams.csv")
#     if os.path.exists(outFile):
#         os.remove(outFile) 
#     
#     # Get list of stop words (will be used later)
#     stopwords = set(stopwords.words(stopword_language)) 
#     
#     stemmer = SnowballStemmer(stopword_language) 
#     
#     recordCounter = 0
#     spec_char_rep_pat = r'[^a-zA-Z0-9\s]|\n'
#     # Open the input csv file. Loop through each record and process each field (text_fields)
#     csv.field_size_limit(sys.maxsize)
#     with open(text_file, mode='r') as csvFile:
#         
#         # Create the sentiment analyzer for some basic sentiment tests.
#         nltkSentiment = SentimentIntensityAnalyzer()
#         
#         csvReader = csv.DictReader(csvFile)
#         lineCount = 0
#         for csvRow in csvReader:
#             recordID = csvRow[record_id_col]
#     
#             # Process each text field.
#             for textItem in text_fields:
#                 # One column/field of text; a word or sentence, or more:
#                 text = csvRow[textItem]
#     
#                 recordCounter += 1
#                 
#                 # Text cleanup
#                 text = " " + text.lower() 
#                 text = re.sub(spec_char_rep_pat, ' ', text) # Replace all none alphanumeric characters with spaces
#     
#                 # Break into single words
#                 tokens = [token for token in text.split(" ") if token != ""]
#                 output = list(ngrams(tokens, 1))
#     
#                 sectionWordCount = math.ceil(len(output)/numberOfSections)
#     
#                 # Write single words to csv
#                 outFile = os.path.join(outfiles_dir, "words.csv")
#     
#                 with open(outFile,'a', newline='') as out:
#                     csvOut = csv.writer(out)
#     
#                     # Write the heading
#                     if recordCounter == 1:
#                         heading = (f"{recordID},word,stem,stop_word,sentiment,word_number,section,section_Word_Number").split(',')
#                         csvOut.writerow(heading)
#     
#                     wordNumber = 1
#                     wordNumberInSection = 1
#                     section = 1
#     
#                     # Write each word
#                     for row in output:
#                         word = ''.join(row) #  Convert the tuple to a string
#     
#                         # Get the word's stem.
#                         wordStem = stemmer.stem(word)
#     
#                         # Get the word's sentiment score.
#                         score = nltkSentiment.polarity_scores(word)
#                         compoundScore = score['compound']
#     
#                         if word in stopwords:
#                             isStopWord = True
#                         else:
#                             isStopWord = False
#     
#                         row = (str(textItem),) + (str(recordID),) + row + (wordStem,) + (str(isStopWord),) + (str(compoundScore),) + (str(wordNumber),) + (str(section),) + (str(wordNumberInSection),)
#                         csvOut.writerow(row)
#     
#                         # Update counter and section.
#                         if wordNumberInSection % sectionWordCount == 0:
#                             section = section + 1
#                             wordNumberInSection = 1
#                         else:
#                             wordNumberInSection = wordNumberInSection + 1
#     
#                         wordNumber = wordNumber + 1
#     
#                 # Parse into n-grams
#                 tokens = [token for token in text.split(" ") if token != ""]
#                 output = list(ngrams(tokens, ngram_len))
#     
#                 # Write n-grams to csv
#                 outFile = os.path.join(outfiles_dir,"NGrams.csv")
#     
#                 with open(outFile,'a', newline='') as out:
#                     csvOut = csv.writer(out)
#     
#                     # Write the heading
#                     if recordCounter == 1:
#                         heading = (recordID,'Word1',)
#                         for i in range(2, ngram_len+1):
#                             heading = heading + ('Word' + str(i),)
#                         
#                         heading = heading + ('Full N-Gram','N-Gram Sentiment','N-Gram Number','Section','Section N-Gram Number')
#                         csvOut.writerow(heading)
#     
#                     wordNumber = 1
#                     wordNumberInSection = 1
#                     section = 1
#     
#                     # Write each n-gram
#                     fullLine = ''
#                     for row in output:
#                         fullLine = ' '.join(row) # Build the full string with spaces
#                         
#                         # Get the n-gram's sentiment score.
#                         score = nltkSentiment.polarity_scores(fullLine)
#                         compoundScore = score['compound']
#     
#                         row = (str(textItem),) + (str(recordID),) + row + (fullLine,) + (str(compoundScore),) + (str(wordNumber),) + (str(section),) + (str(wordNumberInSection),)
#                         csvOut.writerow(row)
#     
#                         # Update counter and section.
#                         if wordNumberInSection % sectionWordCount == 0:
#                             section = section + 1
#                             wordNumberInSection = 1
#                         else:
#                             wordNumberInSection = wordNumberInSection + 1
#     
#                         wordNumber = wordNumber + 1
#     
#             lineCount += 1
            
# ----------------------------- Main ----------------

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Here is what this package does."
                                     )

    parser.add_argument('-d', '--outdir',
                        help='directory for output; default: same as infile',
                        default=None)
    parser.add_argument('-n', '--ngramlen',
                        type=int,
                        help='Ngram order: How many words per ngram; default 3',
                        default=3)
    parser.add_argument('-i', '--idcol',
                        help="name of column that holds unique id; default: 'id'",
                        default='id')
    parser.add_argument('-l', '--language',
                        help="language of text; default: 'english'",
                        default='english')
    parser.add_argument('input_file',
                        help='CSV input file')
    parser.add_argument('columns',
                        type=str,
                        nargs='+',
                        help="Column names that contain text to be analyzed; space separated list")

    args = parser.parse_args();
    

    if not os.path.exists(args.input_file):
        print(f"Input file '{args.input_file}' does not exist")
        sys.exit(1)

#     text_file = '/tmp/propub_msgs.csv'
#     outfiles_dir = os.path.dirname(text_file)
#     text_fields = ['message']
#     record_id_col = 'id'
    
    TextAnalyzer(args.input_file,
                 args.columns,
                 outfiles_dir=args.outdir,
                 record_id_col=args.idcol,
                 stopword_language=args.language,
                 ngram_len=args.ngramlen
                 )

