from __future__ import print_function
import numpy as np

def chandrupatla(f,x0,x1,verbose=False, 
                 eps_m = None, eps_a = None, 
                 maxiter=80, return_iter=False, args=(),):
    # adapted from an earlier implementation of mine 
    # as written in https://www.embeddedrelated.com/showarticle/855.php
    # which in turn is based on Chandrupatla's algorithm as described in Scherer
    # https://books.google.com/books?id=cC-8BAAAQBAJ&pg=PA95
    # This allows vector arguments for x0, x1, and args
    
    # Initialization
    b = x0
    a = x1
    fa = f(a, *args)
    fb = f(b, *args)
    
    # Make sure we know the size of the result
    shape = np.shape(fa)
    assert shape == np.shape(fb)
        
    # In case x0, x1 are scalars, make sure we broadcast them to become the size of the result
    b += np.zeros(shape)
    a += np.zeros(shape)

    fc = fa
    c = a
    
    # Make sure we are bracketing a root in each case
    assert (np.sign(fa) * np.sign(fb) <= 0).all()
    t = 0.5
    # Initialize an array of False,
    # determines whether we should do inverse quadratic interpolation
    iqi = np.zeros(shape, dtype=bool)
    
    # jms: some guesses for default values of the eps_m and eps_a settings
    # based on machine precision... not sure exactly what to do here
    eps = np.finfo(float).eps
    if eps_m is None:
        eps_m = eps
    if eps_a is None:
        eps_a = 2*eps
    
    iterations = 0
    terminate = False
    
    while maxiter > 0:
        maxiter -= 1
        # use t to linearly interpolate between a and b,
        # and evaluate this function as our newest estimate xt
        xt = a + t*(b-a)
        ft = f(xt, *args)
        if verbose:
            output = 'IQI? %s\nt=%s\nxt=%s\nft=%s\na=%s\nb=%s\nc=%s' % (iqi,t,xt,ft,a,b,c)
            if verbose == True:
                print(output)
            else:
                print(output,file=verbose)
        # update our history of the last few points so that
        # - a is the newest estimate (we're going to update it from xt)
        # - c and b get the preceding two estimates
        # - a and b maintain opposite signs for f(a) and f(b)
        samesign = np.sign(ft) == np.sign(fa)
        c  = np.choose(samesign, [b,a])
        b  = np.choose(samesign, [a,b])
        fc = np.choose(samesign, [fb,fa])
        fb = np.choose(samesign, [fa,fb])
        a  = xt
        fa = ft
        
        # set xm so that f(xm) is the minimum magnitude of f(a) and f(b)
        fa_is_smaller = np.abs(fa) < np.abs(fb)
        xm = np.choose(fa_is_smaller, [b,a])
        fm = np.choose(fa_is_smaller, [fb,fa])
        
        """
        the preceding lines are a vectorized version of:

        samesign = np.sign(ft) == np.sign(fa)        
        if samesign
            c = a
            fc = fa
        else:
            c = b
            b = a
            fc = fb
            fb = fa

        a = xt
        fa = ft
        # set xm so that f(xm) is the minimum magnitude of f(a) and f(b)
        if np.abs(fa) < np.abs(fb):
            xm = a
            fm = fa
        else:
            xm = b
            fm = fb
        """
        
        tol = 2*eps_m*np.abs(xm) + eps_a
        tlim = tol/np.abs(b-c)
        terminate = np.logical_or(terminate, np.logical_or(fm==0, tlim > 0.5))
        if verbose:            
            output = "fm=%s\ntlim=%s\nterm=%s" % (fm,tlim,terminate)
            if verbose == True:
                print(output)
            else:
                print(output, file=verbose)

        if np.all(terminate):
            break
        iterations += 1-terminate
        
        # Figure out values xi and phi 
        # to determine which method we should use next
        xi  = (a-b)/(c-b)
        phi = (fa-fb)/(fc-fb)
        iqi = np.logical_and(phi**2 < xi, (1-phi)**2 < 1-xi)
            
        if not shape:
            # scalar case
            if iqi:
                # inverse quadratic interpolation
                t = fa / (fb-fa) * fc / (fb-fc) + (c-a)/(b-a)*fa/(fc-fa)*fb/(fc-fb)
            else:
                # bisection
                t = 0.5
        else:
            # array case
            t = np.full(shape, 0.5)
            a2,b2,c2,fa2,fb2,fc2 = a[iqi],b[iqi],c[iqi],fa[iqi],fb[iqi],fc[iqi]
            t[iqi] = fa2 / (fb2-fa2) * fc2 / (fb2-fc2) + (c2-a2)/(b2-a2)*fa2/(fc2-fa2)*fb2/(fc2-fb2)
        
        # limit to the range (tlim, 1-tlim)
        t = np.minimum(1-tlim, np.maximum(tlim, t))
        
    # done!
    if return_iter:
        return xm, iterations
    else:
        return xm
