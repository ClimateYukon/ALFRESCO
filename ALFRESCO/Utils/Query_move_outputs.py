#This script is used to query outputs files, when using the full shapefile, the amount of folders/plots is too high for manual exploration

import os,glob
path = '/atlas_scratch/jschroder/ALF_outputs/PP_2017-04-26-16-49/'
import shutil

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.isdir(src):
       shutil.copy2(src, dst) 
    else :
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)


models = [os.path.basename(i) for i in glob.glob(os.path.join(path,'Plots_SERDP' ,'*'))]

for model in models :
    print model
    l = glob.glob(os.path.join(path, 'Plots_SERDP', model,"*Statewide*"))
    out = os.path.join(path,"sorted",model,'plots')

    if not os.path.exists( out ) :
        os.makedirs( out )
    _ = [copytree(f, out) for f in l]


#in csv
for metric in metrics :
    print metric
    l = glob.glob(os.path.join(path ,'CSV', metric,'*Statewide*csv'))
    for f in l :
        model = '_'.join(f.split("_")[-4:-2])
        out = os.path.join(path,"sorted",model,'csv',metric)
        if not os.path.exists( out ) :
            os.makedirs( out )
        _ = shutil.copy(f, out)

# for i in glob.glob(os.path.join(path,'sorted','*')):                      
#     shutil.make_archive(os.path.join(path,'sorted', os.path.basename(i)), 'zip', i)
shutil.make_archive(os.path.join(path,'sorted', "Delivery"), 'zip', os.path.join(path,'sorted'))
