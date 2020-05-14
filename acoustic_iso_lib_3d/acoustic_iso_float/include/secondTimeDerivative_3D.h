#ifndef SECOND_TIME_DERIVATIVE_3D_H
#define SECOND_TIME_DERIVATIVE_3D_H 1

#include "operator.h"
#include "float2DReg.h"

using namespace SEP;

class secondTimeDerivative_3D : public Operator<SEP::float2DReg, SEP::float2DReg>
{
	private:

		int _nt;
		float _dt2;

	public:

		/* Overloaded constructor */
		secondTimeDerivative_3D(int nt, float dt);

		/* Destructor */
		~secondTimeDerivative_3D(){};

  		/* Forward / Adjoint */
  		virtual void forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const;
 		virtual void adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const;

};

#endif
