import preprocessing
import k_means
import wordcloudy

def main():
    # for year in range(2003, 2004):
        preprocessing.main(all=True)
        k_means.main(all=True)
        wordcloudy.main(all=True)

if __name__ == '__main__':
    main()
