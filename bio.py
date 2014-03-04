# Based on answer 2 here:
# http://stackoverflow.com/questions/17409107/obtaining-data-from-pubmed-using-python

import os.path
import sys

import Bio.Entrez
from Bio.Entrez import efetch, read

## CHANGE THIS!! ##
Bio.Entrez.email = 'rhyswat@gmail.com'

def fetch_abstract(pmid):
    """Pass in an article id."""
    pmid = str(pmid)
    try:
        handle = efetch(db='pubmed', id=pmid, retmode='xml')
        xml_data = read(handle)[0]
        article = xml_data['MedlineCitation']['Article']
        abstract = article['Abstract']['AbstractText'][0]
        return abstract
    except KeyError :
        return '<CANNOT UNDERSTAND THE DATA RETURNED>'
    except Bio.Entrez.Parser.NotXMLError :
        return '<BAD XML>'
    except IndexError:
        return None


if __name__ == '__main__' :
    ids = [10024335, 10027665, 10027935, 10028936,
           10029645,10029788,10030325,10047639,
           10049657,10051289,10053176,10063787,
           10066961,10067800,10069777,10072218,
           10073748,10073783,10073846,10075143]
           
    def fetch(pubmed_id, f=sys.stdout) :
        f.write('\n')
        f.write('Abstract for pubmed ID {}\n'.format(i))
        a = fetch_abstract(str(i))
        if a is None :
            f.write('<NOTHING RETURNED>\n')
        else :
            f.write(a)
            f.write('\n')

    output = os.path.join(os.path.dirname(__file__), 'output.txt')
    with open(output, 'w') as f :
        for i in ids :
            print ''
            print 'Fetching ID',i
            fetch(i, f=f)

