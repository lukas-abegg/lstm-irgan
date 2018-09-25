import ftplib
import getopt
import gzip
import io
import os
import re
import signal
import sys
import tarfile
import threading
import time
import zipfile

from contextlib import closing


def usage(name):
    print('''
Usage: python {0} -i file_list.txt.gz -o DIRECTORY [OPTIONS]

OPTIONS
  -h, --help:       Print this usage message
  --only-xml:       Ignore images and other non-textual media
  --ignore-errors:  Continue download on error instead of aborting
'''.format(name))


def log(f):
    '''Log a completed article'''

    global count
    count += 1;
    print('{0:>{1}}/{2}: {3}'.format(count, width, total, f))


def skip(f):
    '''Skip an article that is no longer in the OAS'''

    global count
    count += 1;
    print('{0}> {1} is no longer available'.format('-' * (2 * width + 1), f))


def info(message):
    '''Log a debugging message'''

    print('{0}> {1}'.format('-' * (2 * width + 1), message))


def connect():
    '''Connect to the PMC OAS FTP server'''

    info('Connecting to ftp.ncbi.nlm.nih.gov')
    global pmc
    try:
        pmc = ftplib.FTP('ftp.ncbi.nlm.nih.gov')
        pmc.login()
        pmc.cwd('/pub/pmc')
    except Exception as e:
        print(e)
        abort(1)
    heart_beat()


def disconnect():
    '''Disconnect from the PMC OAS FTP server'''

    info('Disconnecting from ftp.ncbi.nlm.nih.gov')
    global pmc
    pmc.close()
    heart_attack()


def reconnect():
    '''
    Disconnect and then reconnect to the PMC OAS FTP server. This is sometimes
    required because the server can intermittently throw 550 errors, indicating
    that a legitimate file does not exist. When this happens, we reconnect to the
    server and try again.
    '''

    time.sleep(10)
    disconnect()
    time.sleep(10)
    connect()


def process():
    '''
    Connect to the PubMed Central Open Access Subset FTP service and attempt to
    download and extract each archive listed in file_list.txt.gz
    '''

    global pmc

    connect()
    update()

    with closing(gzip.open(in_file, 'rb')) as f:

        # The first line of file_list.txt.gz is the date it was created, so we can
        # skip it here because it's not a file name.
        f.readline()

        for line in f:
            archive = line.strip().split('\t')[0]
            pmcid = line.strip().split('\t')[-1][3:]
            path = archive.split('/')
            directory = os.path.join(out_file, *path[0:-1])

            # If a directory for the article we are going to download already exists
            # in the file system, we have already downloaded the article, perhaps
            # from a previous attempt, and we can skip it here.
            if (extras and os.path.exists(os.path.join(directory, pmcid))) or \
                    (not extras and os.path.exists(os.path.join(directory, \
                                                                '{0}.nxml'.format(pmcid)))):
                log(archive)
                continue

            if archive not in archives:
                skip(archive)
                continue

            # Download and extract the article
            extract(archive, directory, pmcid)

    disconnect()


def files_to_extract(members, pmcid):
    '''Generates the files we want to extract from each archive'''

    for m in members:
        if extras or os.path.splitext(m.name.lower())[1] == '.nxml':
            yield rename_path(m, pmcid)


def rename_path(m, pmcid):
    '''
    Rename files in the archive according to the articles PMCID. We rename
    top-level article directories as well as all *.nxml and *.pdf files to be the
    PMCID. All other files, which include all graphics, vidios, and licenses are
    not renamed.
    '''

    pieces = m.name.split('/')
    pieces[0] = pmcid
    for p in range(1, len(pieces)):
        pieces[p] = re.sub(r'.*(\.nxml|\.pdf)', '{0}\g<1>'.format(pmcid), pieces[p])
    m.name = os.path.join(*pieces) if extras else os.path.join(*pieces[1:])
    return m


def extract(archive, directory, pmcid):
    '''
    Download and extract an article archive from the PMC OAS FTP Service. We make
    a maximum of four attempts to download an archive before giving up. If we
    have to give up, we either skip the file and continue downloading or abort,
    depending on the --abort-on-error command line flag.

    archive   -- the name of the file to download
    directory -- the local directory in which to store the file
    '''

    global requests, pmc

    if not os.path.exists(directory):
        os.makedirs(directory)

    response = io.BytesIO()
    tar = None
    try:
        attempt = 0
        while True:
            attempt += 1
            try:
                requests += 1
                pmc.retrbinary("RETR " + archive, response.write, 32)
                tar = tarfile.open(fileobj=io.BytesIO(response.getvalue()), mode="r:gz")
                tar.extractall(path=directory, members=files_to_extract(tar, pmcid))
                break
            except Exception as e:
                info('{0}, attempt {1}/5'.format(e, attempt))
                reconnect()
                if attempt > 4:
                    raise e
    except Exception as e:
        if not ignore:
            info('{0}, aborting...'.format(e))
            abort(1)
        else:
            info('{0}, ignoring...'.format(e))
            rest()
            return
    finally:
        response.close()
        if tar:
            tar.close()

    log(archive)
    rest()


def update():
    '''
    Download the latest file_list.txt from the PMC OAS FTP service, and store the
    list of files. Sometimes articles get removed from open access, so we don't
    want to spend time trying to download them if they are no longer available.
    '''

    global pmc, archives

    info('Updating list of available articles')

    response = io.BytesIO()
    try:
        attempt = 0
        while True:
            attempt += 1
            try:
                pmc.retrbinary("RETR file_list.txt", response.write, 32)
                break
            except Exception as e:
                info('{0}, attempt {1}/5'.format(e, attempt))
                reconnect()
                if attempt > 4:
                    raise e

        response.seek(0)
        for line in response:
            archives.add(line.strip().split('\t')[0])

    except Exception as e:
        info('{0}, aborting...'.format(e))
        abort(1)
    finally:
        response.close()


def rest():
    '''
    We have to be nice to the PMC servers or we may be blocked. The pause here
    ensures that we are making no more than around 3 requests per second.
    Decrease the length of the pause at your own risk.
    '''

    time.sleep(delay)


def input_count(file_list):
    '''
    Returns the number of lines in the intput file and the width of number
    representing that value. This is only used for logging.
    '''

    count = 0
    with closing(gzip.open(file_list, 'rb')) as f:
        for line in f:
            count += 1
    count -= 1
    lines = count

    width = 0
    while count > 0:
        count = int(count / 10)
        width += 1

    return (lines, width)


def sigint_handler(signal, frame):
    abort(0)


def abort(code):
    global pmc
    if pmc:
        disconnect()
    sys.exit(code)


def heart_beat():
    '''
    Interrupt the download every 10 seconds to determine how many PMC requests we
    are making. Adjust the delay factor to ensure that we aren't making more than
    around 3 represts per second.
    '''

    global drum, requests, delay
    if requests > 0:
        adjustment = (1.0 / 3.0 - (1.0 / (requests / 10.0))) / 2.0
        delay = max(0, delay + adjustment)
        requests = 0
    drum = threading.Timer(10, heart_beat)
    drum.start()


def heart_attack():
    '''
    Disable the timer thread and reset the delay factor to a large value so that
    if we recover from an error, we don't overload the server
    '''

    global drum, delay
    if drum:
        drum.cancel()
    delay = 5.2


if __name__ == '__main__':

    in_file = None  # file_list.txt.gz
    out_file = None  # Directory to save the articles
    drum = None  # Timer thread
    pmc = None  # Connection to PMC OAS FTP
    extras = True  # Also download images?
    ignore = False  # Continue on errors?
    requests = 0  # Number of PMC requests made per time interval
    delay = 0.3  # Starting delay factor
    archives = set()  # Current archives in the OAS

    # Register signal handler
    signal.signal(signal.SIGINT, sigint_handler)

    # Parse command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:o:',
                                   ['help', 'input=', 'output=', 'only-xml', 'ignore-errors'])
    except getopt.GetoptError:
        usage(sys.argv[0])
        abort(1)

    for opt, arg, in opts:
        if opt in ('-h', '--help'):
            usage(sys.argv[0])
            abort(0)
        elif opt in ('-i', '--input'):
            in_file = arg
        elif opt in ('-o', '--output'):
            out_file = arg
        elif opt == '--only-xml':
            extras = False
        elif opt == '--ignore-errors':
            ignore = True

    if not in_file or not out_file:
        usage(sys.argv[0])
        abort(1)

    # Gather information about file_list.txt.gz for logging
    total, width = input_count(in_file)
    count = 0

    # Process file_list.txt.gz
    process()