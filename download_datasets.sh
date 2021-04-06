if [ ! -d "datasets" ]
then
	mkdir datasets
fi

cd datasets

if [ ! -d "Caltech101Silhouettes" ]
then 
mkdir Caltech101Silhouettes
fi
cd Caltech101Silhouettes

if [ ! -e "caltech101_silhouettes_28_split1.mat" ]
then 
	wget https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat
fi

cd ..

if [ ! -d "Freyfaces" ]
then
	mkdir Freyfaces
fi

cd Freyfaces

if [ ! -e "freyfaces.pkl" ]

then
wget https://github.com/y0ast/Variational-Autoencoder/raw/master/freyfaces.pkl

fi

cd ..

if [ ! -d "HistopathologyGray" ]
then
	mkdir HistopathologyGray
fi

cd HistopathologyGray

if [ ! -e "histopathology.pkl.tar.gz" ]

then 

	wget https://github.com/jmtomczak/vae_householder_flow/raw/master/datasets/histopathologyGray/histopathology.pkl.tar.gz

	tar -xvzf histopathology.pkl.tar.gz
fi

cd ..

if [ ! -d "MNIST_static" ]

then 
	mkdir MNIST_static
fi
cd MNIST_static

if [ ! -e "binarized_mnist_train.amat" ]
then 
	wget http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat
	wget http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat
	wget http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat
fi

cd ..

if [ ! -d "OMNIGLOT" ]
then 
	mkdir OMNIGLOT 
fi

cd OMNIGLOT  

if [ ! -e 'chardata.mat' ]
then

 wget https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat

fi 

cd ../../
