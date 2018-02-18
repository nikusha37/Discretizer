class Discretizer:
    '''Discretizes series intelligently by using the target variable
    
    Parameters:
    -----------------------------------------------------------------
       DF : pd.DataFrame of X and Y with shape [n_samples,2]
       maximum_iterations : int, indicates number of binning results. 
           Note that this parameter has to be much less than 
           number of unique records in X series
       ordered : Bool, whether the X series is ordered predictor or not.
    '''
    
    def __init__(self, df, maximum_iterations = 10, ordered = True):
        self.ordered = ordered
        self.max_iter = maximum_iterations
        self.data = df.copy()
        self.ddata = self.data.dropna()
        self.xname = self.data.columns[0]
        self.yname = self.data.columns[1]
        self.dX = self.ddata.iloc[:,0]
        self.dY = self.ddata.iloc[:,1]
        self.X = self.data.iloc[:,0]
        self.Y = self.data.iloc[:,1]
        self.missing = (self.data.isnull().sum().sum() > 0)
        
        if ordered:
            self.hist, self.edges = np.histogram(self.dX, bins = 'doane')
            self.edges = list(self.edges)
            self.edges[0], self.edges[-1] = -np.inf, np.inf
            self.nancount = self.X.isnull().sum()
            self.nanperc = self.Y[self.X.isnull()].mean()
        else:
            self.data.loc[:,self.xname] = self.data.loc[:,self.xname].astype('str')
    
    def ordered_binning(self, iteration):
        
        if iteration == 0:
            self.table = pd.cut(self.dX, bins = self.edges).to_frame('BINNED')
            self.bins.append(self.edges)
        else:
            self.table = pd.cut(self.dX, bins = self.new_bins).to_frame('BINNED')
            self.bins.append(self.new_bins)
        self.table['TARGET'] = self.dY
        self.group = self.table.groupby('BINNED')['TARGET']
        both = self.group.count()
        bad = self.group.apply(np.count_nonzero)
        good = (both - bad)
        db = bad/bad.sum()
        dg = good/good.sum()
        self.table1 = pd.concat([both,bad,good,db,dg],axis=1).reset_index().sort_values('BINNED')
        self.table1.columns = ['binned','both','bad','good','db','dg']
        self.table1['ratio'] = self.table1.bad/self.table1.both
        self.table1['ratio_diff'] = self.table1.ratio.diff().abs()
        
        wue = np.log((dg/db).replace(0,0.0000000000000001)).to_frame('WOE')
        iv = ((dg-db)*wue.WOE).to_frame('IV')
        self.W = wue.WOE.sum()
        self.iv = iv.IV.sum()
        
        # Identify and merge nearest edges
        argmin1 = self.table1.ratio_diff.idxmin() -1
        argmin2 = self.table1.ratio_diff.idxmin()
        cand1 = self.table1.binned.loc[argmin1]
        cand2 = self.table1.binned.loc[argmin2]
        dropped = self.table1.binned.cat.remove_categories([cand1,cand2])

        self.new_bins = set()
        for i in dropped.dropna().values:
            self.new_bins.update([i.left,i.right])
        self.new_bins = list(self.new_bins)
        self.new_bins.sort()         
    
    def categorical_binning(self, iteration):
        
        table = self.data.groupby(self.xname).agg({self.yname:[len,np.sum,np.mean]}).reset_index()
        table = table.T.reset_index().drop('level_0',axis=1).set_index('level_1').T
        table.columns = ['BIN','COUNT','BAD','PROB']
        table['GOOD'] = (table.COUNT - table.BAD).astype(float)
        table['DB'] = (table.BAD/table.BAD.sum()).astype(float)
        table['DG'] = (table.GOOD/table.GOOD.sum()).astype(float)
        table['WOE'] = np.log((table.DG/table.DB).replace(0,0.0000000000000001))
        table['IV'] = (table.DG-table.DB)*table.WOE
        table['DIFF'] = table.IV.diff().abs()
        self.W = table.WOE.sum()
        self.iv = table.IV.sum()
        self.ivs.append(self.iv)
        self.bins.append(table.BIN.tolist())
        
        cand1 = table.DIFF.idxmin()-1
        cand2 = cand1 + 1
        self.t = table
        self.tup = (cand1,cand2)
        name1 = table.BIN.loc[cand1]
        name2 = table.BIN.loc[cand2]
        merged = name1 + '__' + name2
        self.data.loc[:,self.xname] = self.data.loc[:,self.xname].replace(name1, merged).replace(name2, merged)

    def compare_binning(self):
        self.ivs = []
        self.bins = []
        if self.ordered:
            for it in range(self.max_iter):
                self.ordered_binning(it)
                self.ivs.append(self.iv) 
        else:
            self.data.loc[:,self.xname] = self.data.loc[:,self.xname].astype('str')
            for it in range(min(self.data.loc[:,self.xname].nunique()-1,self.max_iter)):
                self.categorical_binning(it) 
        
        
        return pd.DataFrame(self.bins).T