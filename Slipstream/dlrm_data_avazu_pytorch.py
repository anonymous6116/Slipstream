import sqlite3 as sq3
import torch
from torch.utils import data
import numpy as np
import math

class AvazuDataset(data.Dataset):

    def __init__(self, db_path, split="train", dup_to_mem=True,
            chunk_size=1000, split_type="random", random_seed=123,
            total_workers=1):

        self.db_conn = sq3.connect(db_path)

        if (dup_to_mem):
            on_disk_conn = self.db_conn
            self.db_conn = sq3.connect(":memory:")

            # copy on disk db into memory
            on_disk_conn.backup(self.db_conn)
            on_disk_conn.close()

        self.db_cursor = self.db_conn.cursor()

        self.total_items = self.db_cursor.execute("""SELECT Count(*) FROM data_cleaned""").fetchone()[0]

        self.m_den = 1
        self.counts = None
        self.n_emb = None

        self.samples_index_lookup = None

        self.chunk_size = chunk_size
        self.data_chunk = None
        self.chunk_range = None
        self.split = split

        #per segment training -- just to start
        indicies = np.arange(self.total_items)

        if (split_type == "random"):

            # !!DANGER!!
            # these two lines must be kept together or
            # train / test could merge
            np.random.seed(random_seed)
            indicies = np.random.permutation(indicies)
            # !!DANGER!!

        #split into five segments 
        # first four are for training, last 1 is for val / test
        indicies = np.array_split(indicies, 5)

        #train on the first 4 segements
        train_indicies = np.concatenate(indicies[:-1])

        #total randomization of the first 4 segments
        train_indicies = np.random.permutation(train_indicies)

        #split the last segment into val and test
        test_indicies = indicies[-1]

        #randomize the last segment and split
        # order is important here
        val_indicies, test_indicies = np.array_split(test_indicies, 2)
        test_indicies = np.random.permutation(test_indicies)
        val_indicies = np.random.permutation(val_indicies)

        #we know the number of cols is small enough
        raw_from_db = list(self.db_cursor.execute("""SELECT count FROM col_counts"""))

        self.m_den = 1  #1 dense feature
        self.counts = np.array(raw_from_db).flatten()
        self.n_emb = len(self.counts)

        print("Avazu opened: sparse = {}, dense = {}".format(self.n_emb, self.m_den))

        if self.split == 'train':
            self.samples_index_lookup = train_indicies
        if self.split == 'val':
            self.samples_index_lookup = val_indicies
        if self.split == 'test':
            self.samples_index_lookup = test_indicies

        self.shuffle()

        # lazy load chunk when needed, initially we have no chunk loaded
        self.chunk_range = (-1,-1)

        # preallocate chunk
        self.data_chunk = [None] * self.chunk_size

    def shuffle(self):

        print("Shuffling indicies...")
        self.samples_index_lookup = np.random.permutation(self.samples_index_lookup)
        #self.load_chunk(0)

    #
    # load [start, start+chunk_size) to local storage
    def load_chunk(self, start):

        print("INFO: Loading chunk @ {}".format(start))

        total_to_convert = self.chunk_size
        end = start + self.chunk_size

        op_count = 20000
        subblock_start = 0
        total_sb = math.ceil(self.chunk_size / op_count)

        general_qmark_string = qmark_string = ", ".join(["?" for i in range(op_count)])

        #do operations in blocks of 20k
        for c in range(total_sb):
            # log every 10 blocks


            qmark_string = None
            if (total_to_convert > op_count):
                subblock_end = subblock_start + op_count
            else:
                subblock_end = subblock_start + total_to_convert

            ids = []
            #print("chunk loading block {} -> {}".format(start+subblock_start,start+subblock_end))
            for ii, target_row in enumerate(self.samples_index_lookup[start+subblock_start:start+subblock_end]):
                sql_raw_row = target_row + 1
                ids.append(int(target_row))

            # if we have a shorter list of ids near the end
            if (len(ids) != op_count):
                qmark_string = ", ".join(["?" for i in range(len(ids))])
            else:
                qmark_string = general_qmark_string

            #break out early if we have no more ids to process
            if (len(ids) == 0):
                break

            query = """SELECT * FROM data_cleaned WHERE rowid IN ({})""".format(qmark_string)

            for ii, data_tuple in enumerate(self.db_cursor.execute(query, ids)):

                #sqlite rows are indexed from 1
                X_int = torch.tensor((data_tuple[2], ), dtype=torch.float)
                X_cat = torch.tensor(data_tuple[3:], dtype=torch.long)
                y = torch.tensor(data_tuple[1], dtype=torch.float)

                self.data_chunk[ii+subblock_start] = (X_int, X_cat, y)

            subblock_start = subblock_end
            total_to_convert -= op_count

        #print("loaded range: {} -> {}".format(start,end))
        self.chunk_range = (start,end)

    def __len__(self):
        return len(self.samples_index_lookup)

    def __getitem__(self, index):

        chunk_start = self.chunk_range[0]
        chunk_end = self.chunk_range[1]

        #test our current chunk
        if isinstance(index, slice):
            if (index.end - index.start) > self.chunk_size:
                raise IndexError("Slice range is larger than chunk size")

            if index.start < chunk_start or index.end >= chunk_end:
                self.load_chunk(index.start)
        else:
            if (index >= chunk_end or index < chunk_start):
                self.load_chunk(index)

        # grab new in case they were updated by the previous
        chunk_start = self.chunk_range[0]
        chunk_end = self.chunk_range[1]

        #correct indexes
        rel_index = None
        if isinstance(index, slice):
            rel_index = slice(index.start-chunk_start, index.end-chunk_end)
        else:
            rel_index = index - chunk_start

        #print(index, self.data_chunk[rel_index])
        return self.data_chunk[rel_index]

def make_avazu_data_and_loaders(args):

    train_data = AvazuDataset(
        args.avazu_db_path,
        split = 'train',
        dup_to_mem = False,
        #chunk_size=3000000
        chunk_size=1
    )
    
    test_data = AvazuDataset(
        args.avazu_db_path,
        split = 'val',
        dup_to_mem = False,
        chunk_size=1
        #chunk_size=20000
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_avazu,
        pin_memory=False,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_avazu,
        pin_memory=False,
        drop_last=False,
    )

    return train_data, train_loader, test_data, test_loader

def load_avazu_preprocessed_data_and_loaders(args, train_data, test_data):

    train_ld = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_avazu,
        pin_memory=False,
        drop_last=False,  # True
    )

    test_ld = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_avazu,
        pin_memory=False,
        drop_last=True,  # True
    )

    return train_ld, test_ld

def load_avazu_test_data_and_loaders(args, test_data):

    test_ld = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_avazu,
        pin_memory=False,
        drop_last=True,  # True
    )

    return test_ld

def collate_wrapper_avazu(list_of_tuples):
    # where each tuple is (X_int, X_cat, y)
    transposed_data = list(zip(*list_of_tuples))
    '''
    print("X_int", transposed_data[0])
    print("X_cat", transposed_data[1])
    print("T", transposed_data[2])
    '''
    X_int = torch.tensor(transposed_data[0])
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T     = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1,1)

    sz0 = X_cat.shape[0]
    sz1 = X_cat.shape[1]

    '''
    if use_gpu and args.num_workers == 0:
        lS_i = [X_cat[:, i].pin_memory() for i in range(sz1)]
        lS_o = [torch.tensor(range(sz0)).pin_memory() for _ in range(sz1)]
        return X_int.pin_memory(), lS_o, lS_i, T.pin_memory()
    else:
        lS_i = [X_cat[:, i] for i in range(sz1)]
        lS_o = [torch.tensor(range(sz0)) for _ in range(sz1)]
        return X_int, lS_o, lS_i, T
    '''
    
    
    lS_i = [X_cat[:, i] for i in range(sz1)]
    lS_o = [torch.tensor(range(sz0)) for _ in range(sz1)]
    return X_int, torch.stack(lS_o), torch.stack(lS_i), T

