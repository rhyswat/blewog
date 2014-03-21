# Based on answer 2 here:
# http://stackoverflow.com/questions/17409107/obtaining-data-from-pubmed-using-python

import io
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
    except Exception as e :
        return '{}: {}'.format(e.__class__.__name__, e)


if __name__ == '__main__' :
    ids = [10024335, 10027665, 10027935, 10028936,
           10073748,10073783,10073846,10075143,
           7641605, 'ericthered']
           
    def fetch(pubmed_id, f=sys.stdout) :
        f.write(u'\n')
        f.write(u'Abstract for pubmed ID {}\n'.format(i))
        a = fetch_abstract(str(i))
        if a is None :
            f.write(u'<NOTHING RETURNED>\n')
        else :
            f.write(unicode(a))
            print a
            f.write(u'\n')

    output = os.path.join(os.path.dirname(__file__), 'output.txt')
    with io.open(output, 'w', encoding='utf-8') as f :
        for i in ids :
            print ''
            print 'Fetching ID',i
            fetch(i, f=f)

