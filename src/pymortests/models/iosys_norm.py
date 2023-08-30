import pytest

from pymortests.models.iosys_add_sub_mul import get_model


@pytest.mark.parametrize('parametric', [False, True])
def test_hinf_norm(parametric):
    m = get_model('LTIModel', 0, parametric)

    if parametric:
        mu = m.parameters.parse([-1.])
    else:
        mu = None

    hinf_norm = m.hinf_norm(mu=mu)
    hinf_norm_f, fpeak = m.hinf_norm(mu=mu, return_fpeak=True)

    assert hinf_norm == hinf_norm_f
