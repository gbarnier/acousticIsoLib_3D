#ifndef DSO_GPU_3D_H
#define DSO_GPU_3D_H 1

#include "operator.h"
#include "double5DReg.h"
#include <vector>
#include <omp.h>

using namespace SEP;

class dsoGpu_3D : public Operator<SEP::double5DReg, SEP::double5DReg> {

	private:

        int _nz, _nx, _ny, _nExt1, _nExt2, _hExt1, _hExt2, _fat;
        double _zeroShift;

	public:

		/* Overloaded constructors */
		dsoGpu_3D(int nz, int nx, int ny, int nExt1, int nExt2, int _fat, double zeroShift);

		/* FWD - ADJ */
		void forward(const bool add, const std::shared_ptr<double5DReg> model, std::shared_ptr<double5DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<double5DReg> model, const std::shared_ptr<double5DReg> data) const;

		/* Destructor */
		~dsoGpu_3D(){};

};

#endif
