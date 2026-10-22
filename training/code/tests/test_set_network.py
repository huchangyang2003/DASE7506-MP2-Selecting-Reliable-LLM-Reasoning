"""Check contextual backprop, equivariance, zero start and capacity controls."""
import unittest
import numpy as np
from set_network import initialize, forward, loss_gradients


class SetNetworkTests(unittest.TestCase):
    def examples(self,mode):
        rng=np.random.default_rng(74)
        p=initialize(3,mode,42,dtype=np.float64)
        # Stay away from ReLU boundaries; nonzero output makes every gradient test meaningful.
        p['a'][:]=1.4;p['b'][:]=1.1;p['v'][:]=rng.normal(size=len(p['v']))*0.2
        if 'C' in p:p['C'][:]=rng.normal(size=p['C'].shape)*0.02
        x=rng.normal(size=(2,4,3))*0.03
        y=np.array([[1,0,1,0],[1,1,1,1]],float)
        offset=rng.normal(size=(2,4))
        return p,x,y,offset

    def test_zero_residual_initialization_exactly_preserves_baseline(self):
        rng=np.random.default_rng(14);x=rng.normal(size=(3,4,25)).astype(np.float32);offset=rng.normal(size=(3,4))
        for mode in ('unary16','unary20','set16'):
            np.testing.assert_array_equal(forward(initialize(25,mode,31),x,offset,mode),offset)

    def test_full_network_finite_difference_both_architectures(self):
        for mode in ('unary16','set16'):
            p,x,y,offset=self.examples(mode)
            _,grad,_=loss_gradients(p,x,y,offset,mode)
            for name in p:
                for index in np.ndindex(p[name].shape):
                    old=p[name][index]
                    p[name][index]=old+1e-5;plus=loss_gradients(p,x,y,offset,mode)[0]
                    p[name][index]=old-1e-5;minus=loss_gradients(p,x,y,offset,mode)[0]
                    p[name][index]=old
                    self.assertAlmostEqual((plus-minus)/2e-5,grad[name][index],places=7,msg=f'{mode}:{name}{index}')

    def test_permutation_scores_loss_and_gradients(self):
        p,x,y,offset=self.examples('set16');order=[2,0,3,1]
        expected=forward(p,x,offset,'set16')
        np.testing.assert_allclose(forward(p,x[:,order],offset[:,order],'set16'),expected[:,order],atol=1e-14)
        loss,g,_=loss_gradients(p,x,y,offset,'set16')
        other,og,_=loss_gradients(p,x[:,order],y[:,order],offset[:,order],'set16')
        self.assertAlmostEqual(loss,other,places=13)
        for name in g:np.testing.assert_allclose(g[name],og[name],atol=1e-13)

    def test_equal_parameter_count_and_shared_initial_weights(self):
        solo=initialize(25,'unary16',23);wide=initialize(25,'unary20',23);joint=initialize(25,'set16',23)
        self.assertEqual(sum(v.size for v in solo.values()),705)
        self.assertEqual(sum(v.size for v in wide.values()),961)
        self.assertEqual(sum(v.size for v in joint.values()),961)
        for name in solo:np.testing.assert_array_equal(solo[name],joint[name])
        np.testing.assert_array_equal(joint['C'],0)

    def test_context_has_effect_and_gradient_through_other_candidates(self):
        p,x,y,offset=self.examples('set16')
        p['C'][:]=0.02;p['v'][:]=0.1
        before=forward(p,x,offset,'set16');altered=x.copy();altered[0,1]+=0.2
        after=forward(p,altered,offset,'set16')
        self.assertNotEqual(before[0,0],after[0,0])
        p.pop('C')
        np.testing.assert_array_equal(forward(p,altered,offset,'unary16')[:,0],forward(p,x,offset,'unary16')[:,0])


if __name__=='__main__':unittest.main()
