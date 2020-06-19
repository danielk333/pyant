
def _generate_interpolation_beam_data(fname, beam, res = 1000):
    '''Create a grid of wave vector projections and 2d interpolate the gain function.
    '''
    beam.point(az0=0.0, el0=90.0)

    save_raw = fname.split('.')
    save_raw[-2] += '_data'
    save_raw = '.'.join(save_raw)

    if not os.path.isfile(save_raw):

        kx=np.linspace(-1.0, 1.0, num=res)
        ky=np.linspace(-1.0, 1.0, num=res)
        
        S=np.zeros((res,res))
        Xmat=np.zeros((res,res))
        Ymat=np.zeros((res,res))

        cnt = 0
        tot = res**2

        for i,x in enumerate(kx):
            for j,y in enumerate(ky):
                
                if cnt % int(tot/1000) == 0:
                    print('{}/{} Gain done'.format(cnt, tot))
                cnt += 1

                z2 = x**2 + y**2
                if z2 < 1.0:
                    k=np.array([x, y, np.sqrt(1.0 - z2)])
                    S[i,j]=beam.gain(k)
                else:
                    S[i,j] = 0;
                Xmat[i,j]=x
                Ymat[i,j]=y
        np.save(save_raw, S)

    S = np.load(save_raw)

    f = sio.interp2d(kx, ky, S.T, kind='linear')
    np.save(fname, f)

