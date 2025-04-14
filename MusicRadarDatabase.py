import json
import os

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
import re

import acousticmeasures as amt

import sqlite3
import librosa 

# define a class to access the musicradar database
class MusicRadarDB:
    def __init__(self, path):
        self.path_to_data = path
        self.all_wavefilelist = []
        self.alldata = pd.DataFrame()
        self.metadata_is_necessary = False
        
    def set_metadata_is_necessary(self, isnecessary:bool = True)->None:
        self.metadata_is_necessary = isnecessary

    def get_filepaths(self)->list:
        """This function will generate the file names in a directory 
        tree by walking the tree either top-down or bottom-up. For each 
        directory in the tree rooted at directory top (including top itself), 
        it yields a 3-tuple (dirpath, dirnames, filenames). In Addition a text
        file is created and saved that contains all file names

        Returns
        -------
        list
            contains all files with the ending .wav
        """
        file_ending=".wav"
        self.all_wavefilelist = []
        for root, directories, files in os.walk(self.path_to_data ):
            files = [os.path.join(root, file) for file in sorted(files) if file.endswith(file_ending)]
            self.all_wavefilelist.extend(files)
        
        
        return self.all_wavefilelist
    
    def save_dataframe(self, filename:str = "musicradar_df.csv")->None:
        """This function will save the dataframe to a csv file
        """
        self.alldata.to_csv(filename, index=False)

    def load_dataframe(self, filename:str = "musicradar_df.csv")->None:
        """This function will load the dataframe from a csv file
        """
        self.alldata = pd.read_csv(filename)
        self.all_wavefilelist = self.alldata.Filename.tolist()

    def validate_wavefilelist(self, wavefilelist, update_internal_list = True)->list:
        """This function will validate the wavefilelist and remove all files that are not valid
        """
        valid_wavefilelist = []
        for file in wavefilelist:
            if os.path.isfile(file):
                try:
                    sf.info(file)
                except RuntimeError as e:
                    print(f"File {file} is not a valid wave file: {e}")
                    continue
                valid_wavefilelist.append(file)
            else:
                print(f"File {file} does not exist")
        if update_internal_list:
            self.all_wavefilelist = valid_wavefilelist
            self.alldata = pd.DataFrame(self.all_wavefilelist, columns=["Filename"])

        return valid_wavefilelist
    
    # check for each wave file in the wavefilelist, if the corresponding json file in audiocommons folder, in chroma_analysis and pt_analysis exists
    def check_ifmetadata_exists(self, wavefilelist=[])->list:
        """This function will check if the metadata files exist for each wave file in the wavefilelist
        """
        metadata_files = []
        hasallmetadata = []
        # if wavefilelist is empty 
        if wavefilelist == []:
            wavefilelist = self.alldata["Filename"]


        for file in wavefilelist:
            hasmetadata = 0
            # json file in subdirectory audiocommons
            path_to_file = os.path.dirname(file)
            metadata_file = os.path.join(path_to_file, "audiocommons", os.path.basename(file).replace(".wav", "_analysis.json"))
            if os.path.isfile(metadata_file):
                metadata_files.append(metadata_file)
                # add to pd dataframe
                self.alldata.loc[self.alldata["Filename"] == file, "AudioCommons_Metadata"] = metadata_file
                hasmetadata += 1
            else:
                print(f"Metadata file {metadata_file} does not exist")
            
            # json file in subdirectory chroma_analysis
            metadata_file = os.path.join(path_to_file, "chroma_analysis", os.path.basename(file).replace(".wav", "_chroma.json"))
            if os.path.isfile(metadata_file):
                metadata_files.append(metadata_file)
                # add to pd dataframe
                self.alldata.loc[self.alldata["Filename"] == file, "Chroma_Metadata"] = metadata_file
                hasmetadata += 1
            else:
                print(f"Metadata file {metadata_file} does not exist")

            # json file in subdirectory pt_analysis
            metadata_file = os.path.join(path_to_file, "pt_analysis", os.path.basename(file).replace(".wav", "_pytimbre.json"))
            if os.path.isfile(metadata_file):
                metadata_files.append(metadata_file)
                # add to pd dataframe
                self.alldata.loc[self.alldata["Filename"] == file, "PT_Metadata"] = metadata_file
                hasmetadata += 1
            else:
                print(f"Metadata file {metadata_file} does not exist")

            if hasmetadata == 3:
                hasallmetadata.append(True)
            else:
                hasallmetadata.append(False)
        
        self.alldata["Has_Metadata"] = hasallmetadata 
        self.alldata["Has_Metadata"] = self.alldata["Has_Metadata"].astype(bool)
        # add the metadata files to the dataframe

        return hasallmetadata
    
    def find_patterns_in_filnames(self, pattern)->list:
        # use the pattern list to generate a new subset of filenames, 
        # that contains only the filenames that contain the pattern
        # the pattern list can contain strings or regex patterns
        # if the pattern list is empty, return all filenames
        wavefilelist = self.alldata["Filename"]
        if pattern == []:
            return wavefilelist
        
        # create a new list with the filenames that contain the pattern
        new_list = []

        regex = re.compile(pattern,re.IGNORECASE)
        #search after all matches in all filenames

        for filename in wavefilelist:
            if self.metadata_is_necessary:
                if not self.alldata.loc[self.alldata["Filename"] == filename, "Has_Metadata"].values[0]:
                    continue

            matches = regex.search(filename)
            if matches:
                new_list.append(filename)

        return new_list
    
    # method to download the musicradar database from the internet


    # methods to handle features from the musicradar database
    # add fetures to the database
    # query database by using fetures
    
    # methods to extract features from audio files
    # audio commons
    # chroma analysis
    # pytimbre analysis
    # common infos (fs, channel count, duration, file format)

    # method to convert to sqlite3 database
    # methods to use the sqlite3 database

    # method to compute thumbnails (pictures) from the audio files
    def compute_thumbnails(self, wavefilelist=[]):
        if wavefilelist == []:
            wavefilelist = self.alldata["Filename"]

        max_freq = 8000
        nr_of_bands = int(40)
        blocklength_ms = 50
        overlap_percent = 50

        for file in wavefilelist:
            data, fs = sf.read(file)
            # convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            # check if the data is empty
            if data.size == 0:
                print(f"File {file} is empty")
                continue
            # check if the data is too short
            if data.size < fs * blocklength_ms / 1000:
                print(f"File {file} is too short")
                continue


            blocklen_samples = int(fs * blocklength_ms / 1000)
            # check if even number and change to even number if not
            if blocklen_samples % 2 != 0:
                blocklen_samples += 1
            nfft = int(2 ** np.ceil(np.log2(blocklen_samples)))
            overlap_samples = int(blocklen_samples * overlap_percent / 100)
            window = np.hanning(blocklen_samples)
            hop_samples = int(blocklen_samples - overlap_samples)
            # compute the mel spectrtogram by using librosa
            S = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=nfft, n_mels=nr_of_bands,
                                            fmax=max_freq, hop_length=hop_samples)
            # convert to dB
            S_db = librosa.power_to_db(S, ref=np.max)
            # convert to a numpy array
            S_db = np.array(S_db)
        
            # convert to image min_val = -80, max_val = 0
            min_val = -80
            max_val = 0
            thumbnail = (S_db - min_val) / (max_val - min_val)
            # convert to 0-1 range
            thumbnail = thumbnail * 255
            thumbnail = thumbnail.astype(np.uint8)
            # use colormap viridis to convert to RGB
            # convert to RGB
            # thumbnail = np.stack((thumbnail,) * 3, axis=-1)
            cm = plt.get_cmap('viridis')
            thumbnail = cm(thumbnail)
            # save the thumbnail
            plt.imshow(thumbnail)
            # add new sub-directory thumbnails
            path_to_file = os.path.dirname(file)
            path_to_file = os.path.join(path_to_file, "thumbnails")
            if not os.path.exists(path_to_file):
                os.makedirs(path_to_file)
            
            plt.savefig(os.path.join(path_to_file, os.path.basename(file).replace(".wav", ".png")))
           
            plt.close()



if __name__ == "__main__":
    
    # build path to the musicradar database with os. join

    path = os.path.join(os.path.sep, "media", "bitzer", "T7", "musicradar_copy")
    print(path)
    #path = r"/media/bitzer/T7/musicradar_small/"
    db = MusicRadarDB(path)
    compute_again = False 

    if compute_again:
        all_files = db.get_filepaths()
        all_files = db.validate_wavefilelist(all_files)
        #all_files = db.check_ifmetadata_exists(all_files)
        all_files = db.check_ifmetadata_exists()
        # save the dataframe to a csv file
        db.save_dataframe("musicradar_df.csv")
    else:
        db.load_dataframe("musicradar_df.csv")
        all_files = db.all_wavefilelist

    # pattern = r"(BD)|(kicks)|[^\w]bd[^\w]|_bd_|kick|bass drum|bassdrum|bass-drum|kickdrum"
    # pattern =r"(SD)|(snare)|(rim)"
    '''
    pattern =r"(HH)|(Hat)|(hihat)"
    db.set_metadata_is_necessary(True)
    kickdrums = db.find_patterns_in_filnames(pattern)
    print(kickdrums)

    # save list as csv
    df = pd.DataFrame(kickdrums, columns=["Filename"])
    df.to_csv("highhat.csv", index=False)

    #print(all_files)
    '''

    # load kickdrumslist csv file

    df = pd.read_csv("kickdrums.csv")

    # use the first 10 files and comute the thumbnails
    df = df.head(10)
    # convert to list
    kickdrums = df["Filename"].tolist()
    # compute the thumbnails
    db.compute_thumbnails(kickdrums)

    file_name = '/media/bitzer/T7/musicradar_copy/musicradar-amped-samples/Duke Kit 1 Dub 60bpm C/Dub Drums/dub drums loops/dub drums loop001kick_only001.wav'