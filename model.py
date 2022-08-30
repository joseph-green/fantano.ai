# %%
import pandas as pd
import glob

# %%

    
albums = pd.DataFrame({'spotify_id': pd.Series(dtype='str'),
    'youtube_id': pd.Series(dtype='str'),
    'project_name': pd.Series(dtype='str'),
    'artist': pd.Series(dtype='str'),
    'project_type': pd.Series(dtype='str'),
    'tracks': pd.Series(dtype='int'),
    'project_art': pd.Series(dtype='str'),
    'year': pd.Series(dtype='int'),
    'rating': pd.Series(dtype='int')})


tracks = pd.DataFrame({'spotify_id': pd.Series(dtype='str'),
        'album_id': pd.Series(dtype='str'),
        'youtube_id': pd.Series(dtype='str'),
        'name': pd.Series(dtype='str'),
        'duration': pd.Series(dtype='int'),
        'explicit': pd.Series(dtype='bool'),
        'preview': pd.Series(dtype='str'),
        'key': pd.Series(dtype='int'),
        'mode': pd.Series(dtype='int'),
        'acousticness': pd.Series(dtype='float'),
        'danceability': pd.Series(dtype='float'),
        'energy': pd.Series(dtype='float'),
        'instrumentalness': pd.Series(dtype='float'),
        'liveness': pd.Series(dtype='float'),
        'loudness': pd.Series(dtype='float'),
        'speechiness': pd.Series(dtype='float'),
        'valence': pd.Series(dtype='float'),
        'tempo': pd.Series(dtype='float')})

# %%
import glob 

#recombine the csvs into a dataframe 
RECOMBINE_FILES = True

path = './data' # use your path
track_files = glob.glob(path + "/tracks20*.csv")
album_files = glob.glob(path + "/albums20*.csv")

album_li = []
track_li = []

if RECOMBINE_FILES:
    for filename in album_files:
        df = pd.read_csv(filename, index_col=0, header=0, dtype=dict(albums.dtypes))
        album_li.append(df)

    album_df = pd.concat(album_li, axis=0, ignore_index=True)


    for filename in track_files:
        df = pd.read_csv(filename, index_col=0, header=0,dtype=dict(tracks.dtypes))
        track_li.append(df)

    track_df = pd.concat(track_li, axis=0, ignore_index=True)

    album_df.to_csv('./data/albums.csv')

    track_df.to_csv('./data/tracks.csv')
else:
    album_df = pd.read_csv('./data/albums.csv')
    track_df = pd.read_csv('./data/tracks.csv')

album_df = album_df.sample(frac=1).reset_index(drop=True)
track_df = track_df.sample(frac=1).reset_index(drop=True)
print(album_df)
print(track_df)


# %%
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Concatenate
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from math import floor
from random import sample


# %%




#set up encoders
artists = album_df["artist"].to_numpy().reshape(-1,1)
artist_encoder = OrdinalEncoder().fit(artists)

project_types = album_df["project_type"].to_numpy().reshape(-1,1)
project_type_encoder = OneHotEncoder().fit(project_types)

def create_tracks_input(album_df):
    
    out = []

    for i, row in album_df.iterrows():
        #get all tracks for a single album
        tracks = track_df[track_df['album_id'] == row['spotify_id']] 
        tracks_count = len(tracks)
        for i, track_data in enumerate(tracks.iterrows()):
            
            #print(f"Processing track: {i}/{tracks_count} (id {track_data[0]})")
            track = track_data[1]
            out.append(
                [
                    track["key"],
                    track["mode"],
                    track["acousticness"],
                    track['danceability'],
                    track['energy'],
                    track['instrumentalness'],
                    track['liveness'],
                    track['loudness'],
                    track['speechiness'],
                    track['valence'],
                    track['tempo']
                ]
            )
    return out


def create_input_row(row):
    out = []
    #out.append(row.name) # need to have a row id included to relate to tracks
    out.append(artist_encoder.transform([[row['artist']]])[0][0])
    project_type_encoded = project_type_encoder.transform([[row['project_type']]]).toarray()
    out.extend(project_type_encoded[0])
    out.append(row['year'])
    out.append(int(row['tracks']))
    return out

ratings = to_categorical(album_df["rating"])

album_in = album_df.apply(create_input_row,axis=1,result_type='expand')
#print(album_in)
album_in = MinMaxScaler().fit_transform(album_in)

track_in = create_tracks_input(album_df)
print()
track_in = MinMaxScaler().fit_transform(track_in)



print(len(album_in))
print(len(track_in))



# %%
test_split = 0.2
validation_split = 0.4
test_amt = floor(test_split * len(album_df))

test_ratings = ratings[:test_amt]
test_in = album_in[:test_amt]

print(album_in)
print(track_in)

track_input = Input(shape=(len(track_in[0]),))


album_input = Input(shape=(len(album_in[0]),))
model = Dense(256,activation='relu')(album_input)
model = Dense(256,activation='relu')(model)
output = Dense(len(ratings[0]),activation='softmax')(model)


model = Model(inputs=album_input,outputs=output)
model.summary()
opt = Adam(learning_rate=0.000001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(album_in,ratings,batch_size=32, epochs=25, validation_split=validation_split,shuffle=True,verbose=1)



#evaluate

model.evaluate(test_in,test_ratings)


#output: eleven units for rating 0-10


# %%



