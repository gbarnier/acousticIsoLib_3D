#ifndef DSO_GPU_3D_H
#define DSO_GPU_3D_H 1

#include "operator.h"
#include "float5DReg.h"
#include <vector>
#include <omp.h>

using namespace SEP;

class dsoGpu_3D : public Operator<SEP::float5DReg, SEP::float5DReg> {

	private:

        int _nz, _nx, _ny, _nExt1, _nExt2, _hExt1, _hExt2, _fat;
        float _zeroShift;

	public:

		/* Overloaded constructors */
		dsoGpu_3D(int nz, int nx, int ny, int nExt1, int nExt2, int _fat, float zeroShift);

		/* FWD - ADJ */
		void forward(const bool add, const std::shared_ptr<float5DReg> model, std::shared_ptr<float5DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float5DReg> model, const std::shared_ptr<float5DReg> data) const;

		/* Destructor */
		~dsoGpu_3D(){};

};

#endif
