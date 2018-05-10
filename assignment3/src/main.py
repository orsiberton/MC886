import preprocessing
import k_means
import wordcloudy

def main():
    for year in range(2003, 2004):
        preprocessing.main(year)
        k_means.main(year)
        # wordcloudy.main(year)

if __name__ == '__main__':
    main()
