def splitGzipTestTrainValidation(gzipDatasetFilepath,
                                 nbTestSamples=10000,
                                 nbValSamples=512):  # 264*1):

    # check if the files don't exist
    if not os.path.isfile(gzipDatasetFilepath[:-3] + '_train.gz'):
        # split otherwise
        f_in = gzip.open(gzipDatasetFilepath, 'rb')
        f_out_train = gzip.open(gzipDatasetFilepath[:-3] + '_train.gz', 'wt')
        f_out_test = gzip.open(gzipDatasetFilepath[:-3] + '_test.gz', 'wt')
        f_out_val = gzip.open(gzipDatasetFilepath[:-3] + '_val.gz', 'wt')

        nbTotalSeen = 0
        nbTestSamples_i = 0
        nbValSamples_i = 0
        for line in f_in:
            sentence = line.strip().decode("utf-8")
            if len(sentence) > 0:
                nbTotalSeen += 1

                rand = np.random.rand()
                if rand > .9:
                    if nbTestSamples_i < nbTestSamples:
                        f_out_test.write(sentence + '\r\n')
                        nbTestSamples_i += 1
                elif rand > .8:
                    if nbValSamples_i < nbValSamples:
                        f_out_val.write(sentence + '\r\n')
                        nbValSamples_i += 1
                else:
                    f_out_train.write(sentence + '\r\n')

        f_in.close()
        f_out_train.close()
        f_out_test.close()
        f_out_val.close()

    for set_name in ['data/biased_train.gz', 'data/biased_test.gz', 'data/biased_val.gz']:
        # read sentences in file
        nbLines = 0
        datasetFilename = os.path.join(CDIR, set_name)
        with gzip.open(datasetFilename, mode='r') as f:
            for _ in f:
                nbLines += 1

        print(nbLines, ' were saved in the %s file.' % (set_name))
