package cnn

import "math"

type Optimizer interface {
	Update(params map[string][]float64, grads map[string][]float64) map[string][]float64
}

type Adam struct {
	lr    float64
	beta1 float64
	beta2 float64
	eps   float64
	m     map[string][]float64
	v     map[string][]float64
	t     int
}

func NewAdamOptimizer(lr, beta1, beta2, eps float64) *Adam {
	return &Adam{
		lr:    lr,
		beta1: beta1,
		beta2: beta2,
		eps:   eps,
		m:     make(map[string][]float64),
		v:     make(map[string][]float64),
		t:     0,
	}
}

func (adam *Adam) Update(params map[string][]float64, grads map[string][]float64) map[string][]float64 {
	if len(adam.m) == 0 {
		for k, v := range params {
			adam.m[k] = make([]float64, len(v))
			adam.v[k] = make([]float64, len(v))
		}
	}

	adam.t++

	for k := range params {
		for i := 0; i < len(params[k]); i++ {
			g := grads[k][i]
			adam.m[k][i] += (1 - adam.beta1) * (g - adam.m[k][i])
			adam.v[k][i] += (1 - adam.beta2) * (g*g - adam.v[k][i])
			mHat := adam.m[k][i] / (1 - math.Pow(adam.beta1, float64(adam.t)))
			vHat := adam.v[k][i] / (1 - math.Pow(adam.beta2, float64(adam.t)))
			params[k][i] -= adam.lr * mHat / (math.Sqrt(vHat) + adam.eps)
		}
	}

	return params
}

type noneOptimizer struct {
	lr float64
}

func NewNoneOptimizer(lr float64) Optimizer {
	o := new(noneOptimizer)
	o.lr = lr
	return o
}

func (o *noneOptimizer) Update(params map[string][]float64, grads map[string][]float64) map[string][]float64 {
	for k, v := range params {
		for i := range v {
			params[k][i] -= o.lr * grads[k][i]
		}
	}

	return params
}
