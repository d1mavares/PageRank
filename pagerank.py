import os
import random
import re
import sys
import copy
import pdb
# 4/5/23 version
DAMPING = 0.85
SAMPLES = 100000
THRESH = 1e-3

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    #corpus = crawl('corpus2')
    ranks = sample_pagerank_stack(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    transition = copy.deepcopy(corpus)
    for key in transition:
        if key in corpus[page]:
            transition[key] = (1-damping_factor)/len(corpus) + damping_factor/len(corpus[page])
        else:
            transition[key] = (1-damping_factor)/len(corpus)
    return transition

def sample_pagerank(corpus, damping_factor, n):
    
    #Return PageRank values for each page by sampling `n` pages
    #according to transition model, starting with a page at random.

    #Return a dictionary where keys are page names, and values are
    #their estimated PageRank value (a value between 0 and 1). All
    #PageRank values should sum to 1.
    
    count,pages,cump,cumprob,pagerank = 0,[],[],dict.fromkeys(corpus),dict.fromkeys(corpus)
    for key in pagerank:
        pagerank[key]=0
    # creating the cumprob dict: {page: [cummulative summation of probabilities]}
    # for example, for corpus0:
    # {'4.html': [0.037500000000000006, 0.07500000000000001, 0.9624999999999999, 0.9999999999999999], '1.html': [0.037500000000000006, 0.07500000000000001, 0.9624999999999999, 0.9999999999999999], '2.html': [0.037500000000000006, 0.5, 0.5375, 1.0], '3.html': [0.4625, 0.5, 0.9625, 1.0]}
    # {being at this page: [going to 4, going to 1, going to 2, going to 3]}
    # because in corpus0: {'4.html': {'2.html'}, '1.html': {'2.html'}, '2.html': {'3.html', '1.html'}, '3.html': {'2.html', '4.html'}}
    for pagekey in corpus:
        transition = transition_model(corpus,pagekey,damping_factor)
        for i,key in enumerate(transition):
            cump.append(transition[key])
            if i>0:
                cump[i] += cump[i-1]
        cumprob[pagekey] = cump
        cump = []
    # pages is a list of page names: ['4.html', '1.html', '2.html', '3.html']
    [pages.append(key) for key in corpus]
    # Random initial page
    p = pages[random.randint(0,len(corpus)-1)]
    while count <= n:
        if p == None:
            pass
        p = totempole(random.random(),cumprob[p],pages)
        pagerank[p] += 1
        count += 1
    M = sum(pagerank.values())
    for page in pagerank:
        pagerank[page] /= M
    return pagerank

def sample_pagerank_stack(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # get pages from corpus key/value pairs in dictionary in a list
    pages = list(corpus.keys())
    pagerank = dict()
    # random initial page
    initial_page = random.choice(pages)
    for page in pages:
        pagerank[page] = 0
    # Update probablity
    transition = transition_model(corpus, initial_page, damping_factor)
    for i in range(0, n-1):
        # new page found through the transition model
        random_page = random.choices(list(transition.keys()), list(transition.values()))
        # update probablity
        pagerank[random_page[0]] = pagerank[random_page[0]] + 1/n
        transition = transition_model(corpus, random_page[0], damping_factor)     
    return pagerank

def totempole(dice,cumprob,pages):
    """
    Return the page to go after throwing a dice, 
    knowing the accumulative probability (cumprob) of the current page
    """
    for index,number in enumerate(cumprob):
        if number>dice:
           return pages[index]
    return pages[-1]

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize
    pagerank = dict.fromkeys(corpus)
    pr2 = dict.fromkeys(corpus,.25)
    for key in pagerank:
        # Initialize pagerank as (1-d)/N for all pages
        pagerank[key] = (1-damping_factor)/len(corpus)

    # compute second term of pagerank
    count = 0
    go = 1
    while go:
        count += 1
        for page in pagerank:
            pagerank[page] = (1-damping_factor)/len(corpus)
            for key in corpus:
                if page in corpus[key]:
                    pagerank[page] += damping_factor*pr2[key]/len(corpus[key])
        # Normalizing pagerank
        M = sum(pagerank.values())
        for page in pagerank:
            pagerank[page] /= M

        # From the community:
        #Here are the new_page_ranks values for corpus0 after 1st iteration. Use them to compare to your calculations:
        #{'1.html': 0.14375, '2.html': 0.56875, '3.html': 0.14375, '4.html': 0.14375}
        #Here are the new_page_ranks values for corpus0 after the 2nd iteration: 
        #{'1.html': 0.27921875, '2.html': 0.34296875, '3.html': 0.27921875, '4.html': 0.09859375
        
        # print(f'current iteration: {count}')
        # for page in pagerank:
        #     print(f'current iteration: {page}:{pagerank[page]}')
        go = 0
        for page in pagerank:
#            print([abs(pr2[page]-pagerank[page]),THRESH])
            if abs(pr2[page]-pagerank[page]) >= THRESH:
                pr2 = copy.deepcopy(pagerank)
                go = 1
            if go == 0:
                return pagerank

if __name__ == "__main__":
    main()
