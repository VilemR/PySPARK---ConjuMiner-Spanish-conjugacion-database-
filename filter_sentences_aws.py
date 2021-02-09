import os
from datetime import datetime
from pyspark import SparkContext, SparkFiles
import logging

logger = None

JOB_NAME_PATTERN = "PySpark_Conju_Miner"
JOB_CODE_VERSION = "Ver.1.04 2021-02-08"
EXE_TIME_STAMP = datetime.now()
SENTENCE_LENGTH_MAX = 70                        # Skip sentences longer than 70 (or more/less) chars.


def get_logger(name, level=logging.INFO):
    import logging
    import sys
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        pass
    else:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


logger = get_logger(JOB_NAME_PATTERN, level=logging.DEBUG)
sc = SparkContext(appName=JOB_NAME_PATTERN)  # sc = SparkContext('local[*]')   #for dev phase on localhost

# Path to flat file containing a list of verb forms in the scope of this data mining (verb conjugaciones)
verbs_in_scope_file_path = 's3://conjuminer/unique_verb_forms_top100.txt'  # s3://conjuminer/input/

# Path to folder where are store corpus TXT files (??? GB)
corpus_input_data_path = 's3://conjuminer/input/'

# Path to folder where sentences containing verb forms in scope are dumped to
output_data_scope = 's3://conjuminer/output/' + EXE_TIME_STAMP.strftime("%Y%m%d_%H%M%S")

logger.info("JOB_CODE_VERSION " + JOB_CODE_VERSION)
logger.info("Current folder : " + str(os.path.dirname(os.path.realpath(__file__))))
logger.info(
    "Input file     : " + corpus_input_data_path)
logger.info(
    "Dictionary file: " + verbs_in_scope_file_path)
logger.info("Output file    : " + output_data_scope)

verbs_in_scope = sc.textFile(verbs_in_scope_file_path).map(
    lambda line:
    line.strip().lower()).collect()
corpus_texts = sc.textFile(corpus_input_data_path, use_unicode=True)

"""
This is a filter function checking each sentence of the corpus. If the sentence contains the verb form listed  in the
"verbs_in_scope" and shorter less than SENTENCE_LENGTH_MAX, th sentence is passed into output file.
"""
def contains_verb_in_scope(lower_line):
    lineclean = " " + lower_line.replace(".", " ").replace(",", " ").replace(";", " ").replace(":", " ").replace("?",
                                                                                                                 " ").replace(
        ",", " ") + " "
    if "DIGITO" in lower_line or len(lower_line) > SENTENCE_LENGTH_MAX:
        return
    for verb in verbs_in_scope:
        if (" " + verb + " ") in lineclean:
            yield (verb, lower_line)
    return

"""
These steps are to load input data, split into lines, lines into sentences and those to filter
"""
sentences_in_corpus = corpus_texts.flatMap(lambda line: line.split("."))
sentences_in_corpus = sentences_in_corpus.flatMap(lambda line: contains_verb_in_scope(line))

# This will save ino "one" output file (~200MB, 3.000.000 sentences)
sentences_in_corpus.coalesce(1, True).saveAsTextFile(output_data_scope)

logger.info("Corpus volume (sentences)   : " + str(sentences_in_corpus.count()))
logger.info("Number of verbs in scope    : " + str(len(verbs_in_scope)))

sc.stop()
