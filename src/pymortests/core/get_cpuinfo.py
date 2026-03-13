import warnings

from cpuinfo import get_cpu_info


# get cpu information and print out a the collected info as a warning
def test_show_cpuinfo():
    dct = get_cpu_info()
    keys = ['arch', 'brand_raw', 'vendor_id_raw']
    info = '[CPUINFO] ' + ' '.join(f'{key}: {dct[key]}' for key in keys)
    warnings.warn(info)
