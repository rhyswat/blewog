#!/usr/bin/python2
# Based on answer 2 here:
# http://stackoverflow.com/questions/17409107/obtaining-data-from-pubmed-using-python

import argparse
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

def fetch_and_write(pubmed_id, f=sys.stdout) :
    f.write(u'\n')
    f.write(u'Abstract for pubmed ID {}\n'.format(i))
    a = fetch_abstract(str(i))
    if a is None :
        f.write(u'<NOTHING RETURNED>\n')
    else :
        f.write(unicode(a))
        f.write(u'\n')


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Fetches PubMed abstracts by ID number.')
    parser.add_argument('-i', '--input', default='identifiers.txt',
                        help='input file, one PubMed ID per row, defaults to identifiers.txt')
    parser.add_argument('-o', '--output', default='output.txt',
                        help='the destination output file, defaults to output.txt')
    args = parser.parse_args()
    
    # check that the input file exists
    if not os.path.exists(args.input) :
        print 'Can\'t find input file \'{}\''.format(args.input)
        print 'Please provide the name of a file containing PubMed identifiers.'
        sys.exit(1)

    # confirm overwrite of the output file
    if os.path.exists(args.output) :
        print 'Output file \'{}\' already exists'.format(args.output)
        c = raw_input('Do you want to overwrite it (y/n)? ')
        if not (c=='y' or c=='Y') :
            sys.exit(0)

    # read the input file and strip out blank entries
    with io.open(args.input, 'r', encoding='utf-8') as f :
        identifiers = [x.strip() for x in f]
    identifiers = [x for x in identifiers if len(x) > 0]

    if len(identifiers) == 0 :
        print 'Input file {} contains no ids'.format(args.input)
    else :
        output = os.path.join(os.path.dirname(__file__), args.output)
        with io.open(args.output, 'w', encoding='utf-8') as f :
            for i in identifiers :
                print ''
                print 'Fetching ID',i
                fetch_and_write(i, f=f)
                f.flush()

