cd data

wget -O train.csv.zip 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/6392/train.csv.zip?sv=2015-12-11&sr=b&sig=2Rw73zgTYHhCxfjj4%2FVtJ%2FYAKv0g6i9ittW4wQV34P4%3D&se=2017-05-08T01%3A11%3A31Z&sp=r'
wget -O test.csv.zip 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/6392/test.csv.zip?sv=2015-12-11&sr=b&sig=wItNJUIaprO%2Fn2n%2B%2FLshJ2ymZcn%2BzFNc%2BrzGViXcdMo%3D&se=2017-05-08T01%3A18%3A09Z&sp=r'
wget -O macro.csv.zip 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/6392/macro.csv.zip?sv=2015-12-11&sr=b&sig=u%2F%2FYNkznQhY3yH%2BJAD3mJE%2F1PO6mVMUCDU1Ih3Zx6LU%3D&se=2017-05-08T01%3A21%3A14Z&sp=r'

unzip train.csv.zip test.csv.zip macro.csv.zip
