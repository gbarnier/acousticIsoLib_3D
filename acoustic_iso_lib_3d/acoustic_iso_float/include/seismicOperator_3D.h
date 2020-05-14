#ifndef SEISMIC_OERATOR_3D_H
#define SEISMIC_OERATOR_3D_H 1

#include "interpTimeLinTbb_3D.h"
#include "operator.h"
#include "float2DReg.h"
#include "float3DReg.h"
#include "float4DReg.h"
#include "ioModes.h"
#include "operator.h"
#include "fdParam_3D.h"
#include "deviceGpu_3D.h"
#include "secondTimeDerivative_3D.h"
#include <omp.h>

using namespace SEP;

template <class V1, class V2>
class seismicOperator_3D : public Operator <V1, V2> {

	protected:

		std::shared_ptr<fdParam_3D> _fdParam_3D;
		std::shared_ptr<deviceGpu_3D> _sources, _receivers;
		long long *_sourcesPositionReg, *_receiversPositionReg;
		int _nSourcesReg, _nReceiversReg;
		int _nts;
		int _iGpu, _nGpu, _iGpuId;
		std::shared_ptr<interpTimeLinTbb_3D> _timeInterp_3D;
		std::shared_ptr<secondTimeDerivative_3D> _secTimeDer;
		std::shared_ptr<V2> _sourcesSignals, _sourcesSignalsRegDts, _sourcesSignalsRegDtsDt2, _sourcesSignalsRegDtwDt2, _sourcesSignalsRegDtw;

	public:

		// QC
		virtual bool checkParfileConsistency_3D(std::shared_ptr<V1> model, std::shared_ptr<V2> data) const = 0; // Pure virtual: needs to implemented in derived class

		// Sources
		void setSources_3D(std::shared_ptr<deviceGpu_3D> sources); // This one is for the nonlinear modeling operator
		void setSources_3D(std::shared_ptr<deviceGpu_3D> sources, std::shared_ptr<float2DReg> sourcesSignals); // For the other operators (Born + Tomo + Wemva)

		// Receivers
		void setReceivers_3D(std::shared_ptr<deviceGpu_3D> receivers);

		// Acquisition
		void setAcquisition_3D(std::shared_ptr<deviceGpu_3D> sources, std::shared_ptr<deviceGpu_3D> receivers, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data); // Nonlinear
		void setAcquisition_3D(std::shared_ptr<deviceGpu_3D> sources, std::shared_ptr<float2DReg> sourcesSignals, std::shared_ptr<deviceGpu_3D> receivers, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data); // Born + Tomo

		// Scaling
		void scaleSeismicSource_3D(const std::shared_ptr<deviceGpu_3D> seismicSource, std::shared_ptr<V2> signal, const std::shared_ptr<fdParam_3D> parObj) const;

		// Other mutators
		void setGpuNumber_3D(int iGpu, int iGpuId){_iGpu = iGpu; _iGpuId = iGpuId;}

		// Accessors
		std::shared_ptr<fdParam_3D> getFdParam_3D(){ return _fdParam_3D; }

};

#include "seismicOperator_3D.cpp"

#endif
