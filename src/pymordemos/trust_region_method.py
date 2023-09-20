import numpy as np
from time import perf_counter

def steighaug_toint_truncated_conjugate_gradient_method(g, H, delta, k_max=None, s_0=None, kappa_fgr=0.1, theta=0.5):
    #print('Steihaug Toint truncated conjugate gradient method')
    n = np.shape(g)[0]
    if s_0 is None:
        s_0 = np.zeros(n)
    # if k_max is None:
    #     k_max = n

    s_k = s_0
    g_k = g
    p_k = -g
    k = 0
    while k < k_max:  # np.linalg.norm(g_k) >= np.linalg.norm(g)* min(kappa_fgr, np.linalg.norm(g)**theta): or k < k_max
        k_kappa = np.dot(p_k, np.dot(H, p_k))
        if k_kappa <= 0:
            # berechne positive Nullstelle von ||s_k + sigma p_k|| = delta
            # Ab wann lohnt es sich einen Wert, den ich zweimal brauche,
            # zu speichern?
            Nullstelle = (-np.dot(s_k, p_k) + np.sqrt(np.dot(s_k, p_k) ** 2 +
                                                      np.linalg.norm(p_k) ** 2 * (delta ** 2 - np.linalg.norm(
                s_k) ** 2))) / np.linalg.norm(p_k) ** 2
            s_k_plus_1 = s_k + Nullstelle * p_k
            #print('Lösung des Sub-Problems liegt auf dem Rand, da k_kappa <= 0')
            return s_k_plus_1

        alpha_k = np.linalg.norm(g_k) ** 2 / k_kappa

        value_2 = s_k + alpha_k * p_k  # Variable abspeichern
        if np.linalg.norm(value_2) >= delta:
            # berechne positive Nullstelle sigma von ||s_k + sigma p_k|| = delta
            Nullstelle = (-np.dot(s_k, p_k) + np.sqrt(np.dot(s_k, p_k)** 2 +
                                                      np.linalg.norm(p_k)** 2 * (delta** 2 - np.linalg.norm(s_k) ** 2))) / np.linalg.norm(p_k) ** 2
            s_k_plus_1 = s_k + Nullstelle * p_k
            #print('Lösung des Sub-Problems liegt auf dem Rand, da das Minimum außerhalb der Trust-Region liegt.')
            return s_k_plus_1

        s_k_plus_1 = value_2
        g_k_plus_1 = g_k + alpha_k * np.dot(H, p_k)
        beta_k = np.linalg.norm(g_k_plus_1)**2 / np.linalg.norm(g_k)**2
        p_k_plus_1 = -g_k_plus_1 + beta_k * p_k

        # Werte neu setzen für die Schleife
        s_k = s_k_plus_1
        g_k = g_k_plus_1
        p_k = p_k_plus_1
        k = k + 1
    #print('Lösung des Sub-Problems liegt innerhalb der Trust-Region.')
    return s_k


def trust_region_method(x_0, function, nabla_function, hess_function, delta_0 = 0.1, eta_1=0.1, eta_2=0.9, gamma_1=0.5,
                        gamma_2=0.5, error=10 ** (-6), delta_max=0.75, shrinking_factor = 0.5, k_max = None ):
    assert delta_max >= delta_0, 'Sie müssen entweder ein größeres delta_max oder ein kleineres delta_0 angeben!'

    data = {'optimum': [], 'optimum_value':[], 'optimum_grad': [], 'optimum_hess':[], 'num_evals': 0, 'num_evals_successful' : 0, 'evaluation_points': [], 'delta': [], 'gradient_norm': [], 'time': 0}
    def model_function(x, g, H):
        return np.dot(g, x - x_k) + 0.5 * np.dot(x - x_k, np.dot(H, x - x_k))

    if k_max is None:
        k_max = np.shape(x_0)[0]
    tic = perf_counter()
    x_k = x_0
    delta_k = delta_0
    data['evaluation_points'].append(x_0)
    #Das kann man hier bestimmt noch ein bisschen besser machen an welcher Stelle man g und H ausrechnet.
    gradient_norm = np.linalg.norm(nabla_function(x_k))
    data['gradient_norm'].append(gradient_norm)
    data['delta'].append(delta_k)
    while gradient_norm >= error:
        H = hess_function(x_k)
        g = nabla_function(x_k)

        s_k = steighaug_toint_truncated_conjugate_gradient_method(g=g, H=H, delta=delta_k, k_max = k_max)
        s_k = s_k + x_k  # aufgrund der Transformation der Probleme (S.T.T.C.G.M. löst ein anderes Problem)

        m_k_eva_x_k = model_function(x_k, g, H) #ist eigentlich immer Null
        m_k_eva_s_k_plus_x_k = model_function(s_k, g, H)

        # beachte, dass man das Sub-problem immer ohne die Konstante berechnet..
        # deswegen muss noch function(x_k) dazugerechnet werden. Kürzt sich aber wieder raus wegen m_k(x_k) - m_k(x_k + s_k)
        # m_k_eva_x_k kann eigentlich auch weggelassen werden, da immer Null herauskommt aufgrund der Wahl von m_k
        rho_k = (function(x_k) - function(s_k)) / (m_k_eva_x_k - m_k_eva_s_k_plus_x_k)

        if rho_k >= eta_1:
            #print('der Schritt ist erfolgreich')
            x_k = s_k
            data['evaluation_points'].append(x_k)
            data['num_evals_successful'] += 1
            gradient_norm = np.linalg.norm(nabla_function(x_k))

        #else:
            #print('der Schritt ist nicht erfolgreich')


        #meine Methode das delta_k_plus_1 neu zu setzen mit Hilfe eines shrinkingfactors
        if rho_k >= eta_2:
            delta_k_plus_1 = delta_k * 1/shrinking_factor
            if delta_k_plus_1 > delta_max:
                delta_k_plus_1 = delta_max
        elif eta_1 <= rho_k < eta_2:
            delta_k_plus_1 = delta_k * shrinking_factor
            if delta_k_plus_1 < gamma_2 *delta_k:
                delta_k_plus_1 = gamma_2 * delta_k
        else:
            delta_k_plus_1 = delta_k * shrinking_factor
            if delta_k_plus_1 < gamma_1 * delta_k:
                delta_k_plus_1 = gamma_1 * delta_k
            if delta_k_plus_1 > gamma_2 * delta_k:
                delta_k_plus_1 = gamma_2 * delta_k

        delta_k = delta_k_plus_1

        data['num_evals'] += 1
        #print('x_k', x_k)
        # gradient_norm = np.linalg.norm(nabla_function(x_k))
        data['gradient_norm'].append(gradient_norm)
        data['delta'].append(delta_k)
    data['time'] = perf_counter() - tic
    data['optimum'] = x_k
    data['optimum_value'] = function(x_k)
    data['optimum_grad'] = nabla_function(x_k)
    data['optimum_hess'] = hess_function(x_k)
    return data

def trust_region_method_step(x_k, function, grad_f_eva_x_k, hess_f_eva_x_k, delta_k = 0.1, eta_1=0.1, eta_2=0.9, gamma_1=0.5,
                        gamma_2=0.5, delta_max=0.75, shrinking_factor = 0.5, k_max = None ):
    assert delta_max >= delta_0, 'Sie müssen entweder ein größeres delta_max oder ein kleineres delta_0 angeben!'

    def model_function(x, g, H):
        return np.dot(g, x - x_k) + 0.5 * np.dot(x - x_k, np.dot(H, x - x_k))

    if k_max is None:
        k_max = np.shape(x_0)[0]

    s_k = steighaug_toint_truncated_conjugate_gradient_method(g=grad_f_eva_x_k, H=hess_f_eva_x_k, delta=delta_k, k_max = k_max)
    s_k = s_k + x_k  # aufgrund der Transformation der Probleme (S.T.T.C.G.M. löst ein andere Problem)

    m_k_eva_x_k = model_function(x_k, g, H) #ist eigentlich immer Null
    m_k_eva_s_k_plus_x_k = model_function(s_k, g, H)

    # beachte, dass man das Sub-problem immer ohne die Konstante berechnet..
    # deswegen muss noch function(x_k) dazugerechnet werden. Kürzt sich aber wieder raus wegen m_k(x_k) - m_k(x_k + s_k)
    # m_k_eva_x_k kann eigentlich auch weggelassen werden, da immer Null herauskommt aufgrund der Wahl von m_k
    rho_k = (function(x_k) - function(s_k)) / (m_k_eva_x_k - m_k_eva_s_k_plus_x_k)

    if rho_k >= eta_1:
        #print('der Schritt ist erfolgreich')
        x_k = s_k
        gradient_norm = np.linalg.norm(nabla_function(x_k))
    #else:
        #print('der Schritt ist nicht erfolgreich')

    #meine Methode das delta_k_plus_1 neu zu setzen mit Hilfe eines shrinkingfactors
    if rho_k >= eta_2:
        delta_k_plus_1 = delta_k * 1/shrinking_factor
        if delta_k_plus_1 > delta_max:
            delta_k_plus_1 = delta_max
    elif eta_1 <= rho_k < eta_2:
        delta_k_plus_1 = delta_k * shrinking_factor
        if delta_k_plus_1 < gamma_2 *delta_k:
            delta_k_plus_1 = gamma_2 * delta_k
    else:
        delta_k_plus_1 = delta_k * shrinking_factor
        if delta_k_plus_1 < gamma_1 * delta_k:
            delta_k_plus_1 = gamma_1 * delta_k
        if delta_k_plus_1 > gamma_2 * delta_k:
            delta_k_plus_1 = gamma_2 * delta_k
    return s_k, delta_k


