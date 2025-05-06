import numpy as np
from utils import *
import os
import pandas as pd
import time


class dataset():
    """dataset class for TCN4Flare
    with methods to extract data, GP_fit"""

    def __init__(self, file_path, sub_path = time.strftime('%Y%m%d/'), data_path = '/data/TCN4Flare/dataset/'):
        """初始化数据集类, data_path为数据集所在目录, file_path为sub_path上一级总目录, sub_path为数据集中每个agn文件所在的子目录,一般采用当天日期命名
        例如:file_path = '/data/TCN4Flare/ztf_agns/', sub_path = '20210915/', data_path = '/data/TCN4Flare/dataset/'
        data_path is the directory where the dataset is saved, file_path is the directory where the agn files are located, 
        sub_path is the subdirectory where each agn file is located, and the date is used as the subdirectory name.
        """
        self.file_path = file_path
        self.data_path = data_path
        self.sub_path = sub_path
        os.makedirs(os.path.join(self.data_path, self.sub_path), exist_ok=True)


    def extract_data(self, band):
        """"
        从file_path目录下subpath子目录中提取数据, 将目录中各个csv文件中的AGN对应band的t, mag, mag_err等信息提取
        并将数据整理成numpy数组形式, 返回数据集和agn参数信息, 以及空文件名列表
        数据集raw_dataset的shape为三维数组, 第一维为agn数量, 第二维为时间点数量, 第三维为数据维度, 分别为时间, flux, flux误差
        agn_params的shape为二维数组, 第一维为agn数量, 第二维为agn参数, 分别为agn名称, agn坐标(ra, dec)
        empty_files为空文件名列表, 记录了没有数据的文件名 \n
        extract agn data from the directory(file_path+subpath)
        and organize the data into numpy array format, return the dataset and agn parameters, and empty file name list
        Return 
            raw_dataset, agn_params, empty_files \n
            raw_dataset has a shape of (agn_num, time_num, data_dim), where data_dim is 3, which means time, flux, and flux error \n
            agn_params contains the agn name and coordinates(ra,dec) \n
            empty_files records the file names without data of the band\n
            raw_dataset and agn_params are saved in the 'datapath/subpath/band_raw_dataset.npz' directory, data:dataset, agn_params:agn_params \n
            empty_files is saved in the 'datapath/subpath/band_empty_files.npy' directory \n
        """

        # extarct file names
        files_names = []
        for item in os.scandir(os.path.join(self.file_path, self.sub_path)):
            if item.is_file():
                files_names.append(item.name)

        empty_files = []
        ts = []
        fluxs = []
        flux_errs = []
        raw_agn_params = []

        # extract data from each file
        for file_name in files_names:
            data = pd.read_csv(os.path.join(self.file_path, self.sub_path, file_name))
            data = data.iloc[np.where(data['filtercode'] == band)[0]]
            if len(data) == 0:
                empty_files.append(file_name)
                continue

            agn_name = file_name.split('.')[0]
            agn_ra = data['ra'].values[0]
            agn_dec = data['dec'].values[0]
            raw_agn_params.append([agn_name, agn_ra, agn_dec])

            t = data['mjd'].values
            mag = data['mag'].values
            mag_err = data['magerr'].values
            flux, flux_err = mag2fluxdensity(mag, mag_err)
            
            ts.append(t)
            fluxs.append(flux)
            flux_errs.append(flux_err)

        # pad the data with NaN value to the same length, and stack them into a 3D array
        max_length = max([len(t) for t in ts])
        padded_t = np.array([np.pad(t, (0, max_length - len(t)), constant_values=np.nan) for t in ts])
        padded_flux = np.array([np.pad(flux, (0, max_length - len(flux)), constant_values=np.nan) for flux in fluxs])
        padded_flux_err = np.array([np.pad(flux_err, (0, max_length - len(flux_err)), constant_values=np.nan) for flux_err in flux_errs])

        raw_dataset = np.stack([padded_t, padded_flux, padded_flux_err], axis=2)
        raw_agn_params = np.array(raw_agn_params)

        # save the extracted data and agn parameters
        np.savez_compressed(os.path.join(self.data_path, self.sub_path + band + '_raw_dataset.npz'), raw_dataset=raw_dataset, agn_params=raw_agn_params)
        np.save(os.path.join(self.data_path, self.sub_path + band + '_empty_files.npy'), empty_files)
        
        return raw_dataset, raw_agn_params, empty_files
    
    def GP_fit_raw_dataset(self, band, raw_dataset, raw_agn_params):
        """
        使用celerite库拟合GP模型, 返回GP拟合之后得到的数据集fit_dataset, agn_params(新增2列包含GP模型参数), 拟合失败的agn名称列表 \n
        Utilize celerite library to fit GP models, return the dataset fitted by GP, agn_params(with 2 new columns containing GP model parameters), and the list of agn names that failed to fit GP models \n
        Return
            fit_dataset, agn_params, fail_agns \n
            fit_dataset: 3D numpy array, shape (agn_num, time_num, data_dim), where data_dim is 3, which means time, flux, and flux error \n
            agn_params: 2D numpy array, shape (agn_num, agn_params_num), where agn_params_num is 5, which means agn name, ra, dec, log_sigma, log_rho \n
            fail_agns: 1D numpy array, shape (agn_num), which contains the agn names that failed to fit GP models \n
            fit_dataset and agn_params are saved in the 'datapath/subpath/band_fit_dataset.npz' directory, data:dataset, agn_params:agn_params \n
            fail_agns is saved in the 'datapath/subpath/band_fit_fail_agns.npy' directory \n
        """

        t_preds = []
        flux_preds = []
        flux_err_preds = []
        fit_params = []    # save the fitted GP parameters, including log_sigma, log_rho
        fail_agns= []    # save the agn names that failed to fit GP models

        for i in range(raw_dataset.shape[0]):

            flux = raw_dataset[i,:,1].astype(float)
            flux_err = raw_dataset[i,:,2].astype(float)
            t = raw_dataset[i,:,0].astype(float)

            # remove NaN value
            t = t[~np.isnan(t)]
            flux = flux[~np.isnan(flux)]
            flux_err = flux_err[~np.isnan(flux_err)]

            # fit GP models to the non-flare light curves
            # bin data
            t,flux,flux_err = bin_data(t,flux,flux_err,1,'flux',int(min(t)),int(max(t)+1))

            # remove outliers with 3-sigma rule
            flux_idx = remove_outliers(flux,3)
            flux = np.delete(flux, flux_idx)
            flux_err = np.delete(flux_err, flux_idx)
            t = np.delete(t, flux_idx)

            # GP_fit function to fit GP models to the flare and non-flare light curves
            try:
                model, params_vector = GP_fit(t, flux, flux_err)
                fit_params.append(params_vector)
            except:
                print("GP_fit failed for index", i)
                fit_params.append([-999,-999])
                fail_agns.append(raw_agn_params[i][0])
                continue

            # predict the light curves with 1-day time interval
            t_pred = np.arange(int(np.min(t)), int(np.max(t))+1, 1, dtype=float)
            flux_pred, flux_err_pred = model.predict(flux, t_pred, return_var=True)
            flux_err_pred = np.sqrt(flux_err_pred)
            
            # save the predicted light curves in lists
            t_preds.append(t_pred)
            flux_preds.append(flux_pred)
            flux_err_preds.append(flux_err_pred)
        
        # convert lists to numpy arrays
        fit_params = np.array(fit_params)
        fail_agns = np.array(fail_agns)

        # add the GP parameters to agn_params
        fit_agn_params = np.hstack((raw_agn_params, fit_params))

        # pad the predicted data with NaN value to the same length, and stack them into a 3D array
        max_length = max(len(t_pred) for t_pred in t_preds)
        padded_t_pred = np.array([np.pad(t_pred, (0, max_length - len(t_pred)), constant_values=np.nan) for t_pred in t_preds])
        padded_flux_pred = np.array([np.pad(flux_pred, (0, max_length - len(flux_pred)), constant_values=np.nan) for flux_pred in flux_preds])
        padded_flux_err_pred = np.array([np.pad(flux_err_pred, (0, max_length - len(flux_err_pred)), constant_values=np.nan) for flux_err_pred in flux_err_preds])

        fit_dataset = np.stack([padded_t_pred, padded_flux_pred, padded_flux_err_pred], axis=2)

        # save the fitted dataset and agn parameters
        np.savez_compressed(os.path.join(self.data_path, self.sub_path + band +'_fit_dataset.npz'), data_pred=fit_dataset, agn_params=fit_agn_params)
        np.save(os.path.join(self.data_path, self.sub_path + band + '_fit_fail_agns.npy'), fail_agns)

        return fit_dataset, fit_agn_params, fail_agns


def build_dataset(flare_dataset, noflare_dataset, save_path, save_name, ratio=0.1, num=20000):
    """
    构建训练/测试数据/验证数据集,  ratio为数据集中flare占比, num为数据集总数, save_path为保存路径
    数据集中包含flare和noflare数据, 训练集中flare占比为ratio, noflare占比为1-ratio \n
    build the training/testing/validation dataset, ratio is the ratio of flares in the dataset, num is the total number of data, and save_path is the path to save the dataset
    flare_dataset, nofalre_dataset should be data after GP fit, which means the data have equal time interval.\n
    Return
        train_dataset, train_labels \n
        train_dataset: 3D numpy array, shape (num, time_num, data_dim), where data_dim is 3, which means time, flux, and flux error \n
        train_labels: 1D numpy array, shape (num), which contains the labels of the data, 1 for flare, 0 for noflare \n
        data saved in save_path/{save_name}_{ratio}.npz \n
    """

    # pad the data with NaN value to the same time length
    if flare_dataset.shape[1] > noflare_dataset.shape[1]:
        pad_width = (0, flare_dataset.shape[1] - noflare_dataset.shape[1])
        noflare_dataset = np.pad(noflare_dataset, pad_width=((0,0),pad_width,(0,0)), mode='constant', constant_values=np.nan)
    elif noflare_dataset.shape[1] > flare_dataset.shape[1]:
        pad_width = (0, noflare_dataset.shape[1] - flare_dataset.shape[1])
        flare_dataset = np.pad(flare_dataset, pad_width=((0,0),pad_width,(0,0)), mode='constant', constant_values=np.nan)
    
    # get needed number of flares and noflare data, and combine them into a single dataset
    flares_idx = np.random.choice(flare_dataset.shape[0], int(num*ratio), replace=False)
    data_flares = flare_dataset[flares_idx,:,:]
    noflares_idx = np.random.choice(noflare_dataset.shape[0], int(num*(1-ratio)), replace=False)
    data_noflare = noflare_dataset[noflares_idx,:,:]
    train_dataset = np.vstack((data_flares, data_noflare))

    # shuffle the dataset and get the labels
    indices = np.arange(train_dataset.shape[0])
    np.random.shuffle(indices)
    train_dataset = train_dataset[indices]

    train_labels = np.concatenate((np.ones(int(num*ratio)), np.zeros(int(num*(1-ratio)))))
    train_labels = train_labels[indices]

    # save the dataset and labels
    np.savez_compressed(os.path.join(save_path, f'{save_name}_{ratio}.npz'), data=train_dataset, labels=train_labels)
    
    return train_dataset, train_labels

class FlareDataset(dataset):
    """
    dataset for inserting flares into the dataset, inherited from dataset class
    """
    def __init__(self, file_path, sub_path = time.strftime('%Y%m%d/'), data_path=None):
        super().__init__(file_path, sub_path, data_path)

    def extarct_data_insert_flares(self, band):
        """
        从file_path目录下subpath子目录中提取数据, 将目录中各个csv文件中的AGN对应band的t,mag,mag_err等信息提取
        同时在mag中插入Gamma flare信息,之后转换为flux,并将数据整理成numpy数组形式,返回数据集和agn参数信息,flares参数信息,以及空文件名列表 \n
        extract agn data from the directory(file_path+subpath), insert Gamma flares into the mag data, 
        convert mag to flux, and organize the data into numpy array format, 
        delete LC with less than 100 data points \n
        return the dataset and agn parameters, flares parameters, and empty file name list \n
        Return
            flare_dataset, flare_params \n
            flare_dataset has a shape of (agn_num, time_num, data_dim), where data_dim is 3, which means time, flux, and flux error \n
            flare_params contains the agn name, ra, dec, t_start, t_end, T_dur, peakmag, shape of the flare \n
            flare_dataset and flare_params are saved in the 'datapath/subpath/band_flare_dataset.npz' directory, data:dataset, agn_params:agn_params \n
            empty_files is saved in the 'datapath/subpath/band_empty_files.npy' directory \n
        """

        # extarct file names in the directory
        files = os.listdir(os.path.join(self.file_path, self.sub_path))
        files_names = [file for file in files if file.endswith('.csv')]

        flux_flares = []
        err_flares = []
        t_flares = []
        flare_params = []
        empty_files = []

        # insert flares into the mag data, convert mag to flux, and organize the data into numpy array format
        for file_name in files_names:
            # read the data from csv file
            data = pd.read_csv(os.path.join(self.file_path, self.sub_path, file_name))
            # filter the data by band
            data = data.iloc[np.where(data['filtercode'] == band)[0]]          
            # check if the data is empty
            if len(data) <= 100:
                empty_files.append(file_name)
                continue  
            # get agn name, ra, dec
            agn_name = file_name.split('.')[0]
            agn_ra = data['ra'].values[0]
            agn_dec = data['dec'].values[0]

            # insert flares into the mag data
            t = data['mjd'].values
            mag = data['mag'].values
            mag_err = data['magerr'].values

            # generate random flares' parameters: t_start, peakmag, T_dur, shape
            idx = np.random.randint(int(0.8*len(t)),size=1)[0]
            t_start = t[idx]
            sigma = np.std(mag)
            peakmag = np.random.uniform(max(sigma,0.5),max(sigma,0.5)+1.5)  # induce sigma to avoid flares lower than sigma
            T_dur = np.random.uniform(15,min(300,np.max(t)-t_start),1)[0]
            shape = np.random.choice([2,3,4])
            shape_FWHM = {2:2.44,3:3.4,4:4.13}
            t_end = t_start + 3*np.sqrt(shape*(T_dur/shape_FWHM[shape])**2)      # 3-sigma rule for flare duration
            mag_flare = mag - generate_flare(t,t_start,peakmag,T_dur,shape=shape)

            # convert mag to flux, and organize the data into numpy array format
            flux, flux_err = mag2fluxdensity(mag_flare, mag_err)
            flux_flares.append(flux)
            err_flares.append(flux_err)
            t_flares.append(t)
            flare_params.append([agn_name, agn_ra, agn_dec, t_start,t_end,T_dur,peakmag, shape])

        # convert lists to numpy arrays
        flare_params = np.array(flare_params)

        # pad the flare data with NaN value to the same length, and stack them into a 3D array
        max_length = max([len(t) for t in t_flares])
        padded_t_flares = np.array([np.pad(t_flare, (0, max_length - len(t_flare)), constant_values=np.nan) for t_flare in t_flares])
        padded_flux_flares = np.array([np.pad(flux_flare, (0, max_length - len(flux_flare)), constant_values=np.nan) for flux_flare in flux_flares])
        padded_err_flares = np.array([np.pad(err_flare, (0, max_length - len(err_flare)), constant_values=np.nan) for err_flare in err_flares])

        flare_dataset = np.stack([padded_t_flares, padded_flux_flares, padded_err_flares], axis=2)

        # save the extracted data with human made flares and agn parameters
        np.savez_compressed(os.path.join(self.data_path, self.sub_path + band + '_flare_dataset.npz'), data=flare_dataset,  params=flare_params)
        np.save(os.path.join(self.data_path, self.sub_path + band + '_empty_files.npy'), empty_files)

        return flare_dataset, flare_params