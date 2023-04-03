package cnn

type LRScheduler interface {
	GetLR(epoch int) float64
}

type ConstLR struct {
	lr float64
}

func NewConstLR(lr float64) LRScheduler {
	r := new(ConstLR)
	r.lr = lr
	return r
}

func (r *ConstLR) GetLR(_ int) float64 {
	return r.lr
}

type stepLR struct {
	lr    float64
	step  int
	rate  float64
	minLr float64
}

func NewStepLR(lr, rate, minLr float64, step int) LRScheduler {
	r := new(stepLR)
	r.lr = lr
	r.minLr = minLr
	r.step = step
	r.rate = rate
	return r
}

func (r *stepLR) GetLR(epoch int) float64 {
	if epoch >= r.step && epoch%r.step == 0 && r.lr > r.minLr {
		r.lr = r.lr * (1 - r.rate)
		if r.lr < r.minLr {
			r.lr = r.minLr
		}
	}
	return r.lr
}
