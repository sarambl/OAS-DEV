import pathlib

def get_project_base(hostname, nird_project_code, testmode=False):
    if testmode:
        return str(pathlib.Path(__file__).parent.absolute())
    nird_proj_root = '/projects/'+ nird_project_code+'/sarambl/'
    neptune_proj_root = '/home/ubuntu/mnts/nird/projects/'
    saturn_proj_root = '/home/ubuntu/mnts/nird/projects/'
    path_proj = {'neptune': neptune_proj_root,
             'saturn' : saturn_proj_root,
             'nird': nird_proj_root,
             }
    for i in range(0,5):

        path_proj[f'login{i}-nird-tos'] = nird_proj_root
        path_proj['login-nird-%s' % i] = nird_proj_root
        path_proj['login%s-nird' % i] = nird_proj_root
    return path_proj[hostname]


