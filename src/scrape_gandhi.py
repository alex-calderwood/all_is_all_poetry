import requests
import subprocess
import re, codecs


verbose = True

base_url = 'http://www.gandhiashramsevagram.org/gandhi-literature/mahatma-gandhi-collected-works-volume-{}.pdf'
base_gandhi_file = '../corpus/gandhi/pdf/volume_{}.pdf'
base_gandhi_text = '../corpus/gandhi/text/volume_{}.txt'
base_clean_file = '../corpus/gandhi/text/clean_volume_{}.txt'
gandhi = '../corpus/gandhi/gandhi.txt'

""""
Downloads, cleans and saves the complete works of Gandhi. 

Usage: 
Make sure the folders ../corpus/gandhi/text/ and ../corpus/gandhi/pdf/ exist before you run. Then run:

     python scrape_gandhi.py
"""

def download_pdf(url, save_name):
    response = requests.get(url)

    with open(save_name, 'wb') as f:
        f.write(response.content)


# Download each of the 98 pdfs from the site http://www.gandhiashramsevagram.org/
for i in range(1, 99):
    save_file = base_gandhi_file.format(i)

    if verbose:
        print('Downloading', save_file + '...')

    download_pdf(base_url.format(i), save_file)

# Convert the pdf into a text file
for i in range(1, 99):

    gandhi_text_file = base_gandhi_text.format(i)

    if verbose:
        print('Converting PDF to', gandhi_text_file + '...')

    # Convert the ith pdf into a txt file and save it in the txt/ directory
    result = subprocess.run(['pdftotext', base_gandhi_file.format(i), gandhi_text_file])
    print(result)


def check_regex(regex, text):
    matches = re.finditer(regex, text)

    for match in matches:
        print(match.group(0))
        print('---')


def clean_text():
    # Define the patterns we will match that don't depend on the volume
    collected_works_page_break = re.compile('[\n\s]*[0-9]*[\n\s]*THE COLLECTED WORKS OF.*[\n\s]*\f')
    headers_and_footers = re.compile('\n[a-z|0-9].*[\n\s]+')
    description_of_text = re.compile('[\n\s]*.*photostat.*[\n\s]*')

    for i in range(1, 99):

        # Read the ith text file
        gandhi_text_file = codecs.open(base_gandhi_text.format(i), 'r', 'iso-8859-1')
        text = gandhi_text_file.read()

        # Define the regex that depend on the volume number
        vol_page_break = re.compile('VOL\.\s*{}.*[\n\s]*[0-9]*[\n\s]*\f'.format(i))

        # Successively remove all of the text matched by our patterns
        result1, n1 = re.subn(collected_works_page_break, '', text)
        result2, n2 = re.subn(vol_page_break, '', result1)
        result3, n3 = re.subn(headers_and_footers, '', result2)
        result4, n4 = re.subn(description_of_text, '', result3)

        # For debugging
        #         check_regex(collected_works_page_break, text)

        if verbose:
            print('deletions:', n1, n2, n3, n4)

        clean_text = result4

        with open(base_clean_file.format(i), 'w') as clean_file:
            clean_file.write(clean_text)

        if verbose:
            print('wrote clean file to', base_clean_file.format(i))


clean_text()
print('Done cleaning text')

# Put everything into one big clean file
gandhi_file = open(gandhi, 'w')

for i in range(1, 99):
    clean_text = open(base_clean_file.format(i), 'r').read()
    print(clean_text[:10])
    gandhi_file.write(clean_text)

    print('Wrote', base_clean_file.format(i), 'to', gandhi)

gandhi_file.close()

print('Done.')