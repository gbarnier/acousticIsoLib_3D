#include <vector>
#include <omp.h>
#include "BornExtShotsGpu_3D.h"
#include "BornExtGpu_3D.h"

BornExtShotsGpu_3D::BornExtShotsGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::shared_ptr<SEP::double2DReg> sourcesSignals, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector){

	// Setup parameters
	_par = par;
	_vel = vel;
	_nShot = par->getInt("nShot");
	createGpuIdList_3D();
	_info = par->getInt("info", 0);
	_deviceNumberInfo = par->getInt("deviceNumberInfo", _gpuList[0]);
	assert(getGpuInfo_3D(_gpuList, _info, _deviceNumberInfo)); // Get info on GPU cluster and check that there are enough available GPUs
	_sourcesVector = sourcesVector;
	_receiversVector = receiversVector;
	_sourcesSignals = sourcesSignals;
	// std::cout << "test0" << std::endl;
	// Allocate source wavefields for all GPUs
	// std::cout << "Allocating " << _nGpu << " wavefield(s)" << std::endl;
	axis zAxis = _vel->getHyper()->getAxis(1);
	axis xAxis = _vel->getHyper()->getAxis(2);
	axis yAxis = _vel->getHyper()->getAxis(3);
	axis timeAxis = _sourcesSignals->getHyper()->getAxis(1);
	_srcWavefieldHyper = std::make_shared<hypercube>(zAxis, xAxis, yAxis, timeAxis);
	// std::shared_ptr<SEP::double4DReg> wavefieldTemp;
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		// wavefieldTemp = std::make_shared<SEP::double4DReg>(_srcWavefieldHyper);
		std::shared_ptr<double4DReg> wavefieldTemp(new double4DReg(_srcWavefieldHyper));
		_srcWavefieldVector.push_back(wavefieldTemp);
		_srcWavefieldVector[iGpu]->scale(0.0);
		// std::cout << "[BornExtShotsGpu_3D]: address for source wavefield inside loop [" <<iGpu << "] = " << _srcWavefieldVector[iGpu] << std::endl;
	}

}

void BornExtShotsGpu_3D::createGpuIdList_3D(){

	// Setup Gpu numbers
	_nGpu = _par->getInt("nGpu", -1);

	// Check that the user does not ask for more GPUs than shots to be modeled
	// if (_nGpu > _nShot){std::cout << "**** ERROR [BornExtShotsGpu_3D]: User required more GPUs than shots to be modeled ****" << std::endl; assert(1==2);}

	std::vector<int> dummyVector;
 	dummyVector.push_back(-1);
	_gpuList = _par->getInts("iGpu", dummyVector);

	// If the user does not provide nGpu > 0 or a valid list -> break
	if (_nGpu <= 0 && _gpuList[0]<0){std::cout << "**** ERROR [BornExtShotsGpu_3D]: Please provide a list of GPUs to be used ****" << std::endl; assert(1==2);}

	// If user does not provide a valid list but provides nGpu -> use id: 0,...,nGpu-1
	if (_nGpu>0 && _gpuList[0]<0){
		_gpuList.clear();
		for (int iGpu=0; iGpu<_nGpu; iGpu++){
			_gpuList.push_back(iGpu);
		}
	}

	// If the user provides a list -> use that list and ignore nGpu for the parfile
	if (_gpuList[0]>=0){
		_nGpu = _gpuList.size();
		sort(_gpuList.begin(), _gpuList.end());
		std::vector<int>::iterator it = std::unique(_gpuList.begin(), _gpuList.end());
		bool isUnique = (it==_gpuList.end());
		if (isUnique==0) {
			std::cout << "**** ERROR [BornExtShotsGpu_3D]: Please make sure there are no duplicates in the list ****" << std::endl; assert(1==2);
		}
	}

	// Allocation of arrays of arrays will be done by the gpu # _gpuList[0]
	_iGpuAlloc = _gpuList[0];
}

void BornExtShotsGpu_3D::forward(const bool add, const std::shared_ptr<double5DReg> model, std::shared_ptr<double3DReg> data) const {

	// for (int iGpu=0; iGpu<_nGpu; iGpu++){
	// 	std::cout << "[BornShotGpu]: Forward source wavefield [" <<iGpu << "] = " << _srcWavefieldVector[iGpu] << std::endl;
	// }
	// std::cout << "test2" << std::endl;
	if (!add) data->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (_sourcesSignals->getHyper()->getAxis(2).n == 1) {
			// std::cout << "Constant source signal over shots" << std::endl;
			constantSrcSignal = 1; }
	else {constantSrcSignal=0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1) {
		// std::cout << "Constant receiver geometry over shots" << std::endl;
		constantRecGeom=1;}
	else {constantRecGeom=0;}

	// Create vectors for each GPU
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<BornExtGpu_3D>> BornExtObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create Born object
		// std::cout << "Max source wavefield 1 = " << _srcWavefieldVector[iGpu]->max() << std::endl;
		// std::cout << "Min source wavefield 2 = " << _srcWavefieldVector[iGpu]->min() << std::endl;
		std::shared_ptr<BornExtGpu_3D> BornExtGpuObject(new BornExtGpu_3D(_vel, _par, _srcWavefieldVector[iGpu], _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		BornExtObjectVector.push_back(BornExtGpuObject);
		// std::cout << "Max source wavefield 2 = " << _srcWavefieldVector[iGpu]->max() << std::endl;
		// std::cout << "Min source wavefield 2 = " << _srcWavefieldVector[iGpu]->min() << std::endl;

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			BornExtGpuObject->getFdParam_3D()->getInfo_3D();
		}

		// Allocate memory on device
		allocateBornExtShotsGpu_3D(BornExtObjectVector[iGpu]->getFdParam_3D()->_vel2Dtw2, BornExtObjectVector[iGpu]->getFdParam_3D()->_reflectivityScale, iGpu, _gpuList[iGpu]);

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);


	}

	// exit(0);

	// Launch Born forward
	#pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Case where the source signature is not constant over shots
		std::shared_ptr<SEP::double2DReg> sourcesSignalsTemp;
		axis dummyAxis(1);

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _sourcesSignals, _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			// Create a 2D-temporary array where you store the wavelet for this shot
			sourcesSignalsTemp = std::make_shared<double2DReg>(_sourcesSignals->getHyper()->getAxis(1),dummyAxis);
    		memcpy(sourcesSignalsTemp->getVals(), &(_sourcesSignals->getVals()[iShot*_sourcesSignals->getHyper()->getAxis(1).n]), sizeof(double)*_sourcesSignals->getHyper()->getAxis(1).n);

			// Set the acquisition geometry (shot+receiver) for this specific shot
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], sourcesSignalsTemp, _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _sourcesSignals, _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			// Create a 2D-temporary array where you store the wavelet for this shot
			sourcesSignalsTemp = std::make_shared<double2DReg>(_sourcesSignals->getHyper()->getAxis(1),dummyAxis);
    		memcpy(sourcesSignalsTemp->getVals(), &(_sourcesSignals->getVals()[iShot*_sourcesSignals->getHyper()->getAxis(1).n]), sizeof(double)*_sourcesSignals->getHyper()->getAxis(1).n);
			// Set the acquisition geometry (shot+receiver) for this specific shot
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], sourcesSignalsTemp, _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		BornExtObjectVector[iGpu]->setGpuNumber_3D(iGpu, iGpuId);
		BornExtObjectVector[iGpu]->forward(false, model, dataSliceVector[iGpu]);

		// Store dataSlice into data
		#pragma omp parallel for
		for (int iReceiver=0; iReceiver<hyperDataSlice->getAxis(2).n; iReceiver++){
			for (int its=0; its<hyperDataSlice->getAxis(1).n; its++){
				(*data->_mat)[iShot][iReceiver][its] += (*dataSliceVector[iGpu]->_mat)[iReceiver][its];
			}
		}
	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateBornExtShotsGpu_3D(iGpu, _gpuList[iGpu]);
	}

}

void BornExtShotsGpu_3D::adjoint(const bool add, std::shared_ptr<double5DReg> model, const std::shared_ptr<double3DReg> data) const {

	if (!add) model->scale(0.0);

	// for (int iGpu=0; iGpu<_nGpu; iGpu++){
	// 	std::cout << "[BornShotGpu]: source wavefield address at step #1 of the adjoint for gpu[" <<iGpu << "] = " << _srcWavefieldVector[iGpu] << std::endl;
	// }

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (_sourcesSignals->getHyper()->getAxis(2).n == 1) {
			// std::cout << "Constant source signal over shots" << std::endl;
			constantSrcSignal = 1; }
	else {constantSrcSignal=0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1) {
		// std::cout << "Constant receiver geometry over shots" << std::endl;
		constantRecGeom=1;}
	else {constantRecGeom=0;}

	// Create vectors for each GPU
	std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), model->getHyper()->getAxis(3), model->getHyper()->getAxis(4), model->getHyper()->getAxis(5)));
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double5DReg>> modelSliceVector;
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<BornExtGpu_3D>> BornExtObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create Born object
		std::shared_ptr<BornExtGpu_3D> BornExtGpuObject(new BornExtGpu_3D(_vel, _par, _srcWavefieldVector[iGpu], _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		BornExtObjectVector.push_back(BornExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			BornExtGpuObject->getFdParam_3D()->getInfo_3D();
		}

		// Allocate memory on device for that object
		allocateBornExtShotsGpu_3D(BornExtObjectVector[iGpu]->getFdParam_3D()->_vel2Dtw2, BornExtObjectVector[iGpu]->getFdParam_3D()->_reflectivityScale, iGpu, _gpuList[iGpu]);

		// Model slice
		std::shared_ptr<SEP::double5DReg> modelSlice(new SEP::double5DReg(hyperModelSlice));
		modelSlice->scale(0.0); // Check that
		modelSliceVector.push_back(modelSlice);

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);
	}

	// Launch Born forward
	#pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Case where the source signature is not constant over shots
		std::shared_ptr<SEP::double2DReg> sourcesSignalsTemp;
		axis dummyAxis(1);

		// Copy data slice
		memcpy(dataSliceVector[iGpu]->getVals(), &(data->getVals()[iShot*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n]), sizeof(double)*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n);

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _sourcesSignals, _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {

			// Create a 2D-temporary array where you store the wavelet for this shot
			sourcesSignalsTemp = std::make_shared<double2DReg>(_sourcesSignals->getHyper()->getAxis(1),dummyAxis);
    		memcpy(sourcesSignalsTemp->getVals(), &(_sourcesSignals->getVals()[iShot*_sourcesSignals->getHyper()->getAxis(1).n]), sizeof(double)*_sourcesSignals->getHyper()->getAxis(1).n);

			// Set the acquisition geometry (shot+receiver) for this specific shot
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], sourcesSignalsTemp, _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _sourcesSignals, _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}

		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {

			// Create a 2D-temporary array where you store the wavelet for this shot
			sourcesSignalsTemp = std::make_shared<double2DReg>(_sourcesSignals->getHyper()->getAxis(1),dummyAxis);
    		memcpy(sourcesSignalsTemp->getVals(), &(_sourcesSignals->getVals()[iShot*_sourcesSignals->getHyper()->getAxis(1).n]), sizeof(double)*_sourcesSignals->getHyper()->getAxis(1).n);

			// Set the acquisition geometry (shot+receiver) for this specific shot
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], sourcesSignalsTemp, _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}

		// for (int iGpu=0; iGpu<_nGpu; iGpu++){
		// 	std::cout << "[BornShotGpu]: source wavefield address at step #3 of the adjoint for gpu[" <<iGpu << "] = " << _srcWavefieldVector[iGpu] << std::endl;
		// }

		// Set GPU number for propagator object
		BornExtObjectVector[iGpu]->setGpuNumber_3D(iGpu, iGpuId);
		// BornObjectVector[iGpu]->resetWavefield();
		// Launch modeling
		BornExtObjectVector[iGpu]->resetWavefield();
		// for (int iGpu=0; iGpu<_nGpu; iGpu++){
		// 	std::cout << "[BornShotGpu]: source wavefield address at step #4 of the adjoint for gpu[" <<iGpu << "] = " << _srcWavefieldVector[iGpu] << std::endl;
		// }
		// std::cout << "[BornShotGpu]: max model before = " << modelSliceVector[iGpu]-> max() << std::endl;
		// std::cout << "[BornShotGpu]: min model before = " << modelSliceVector[iGpu]-> min() << std::endl;
		BornExtObjectVector[iGpu]->adjoint(true, modelSliceVector[iGpu], dataSliceVector[iGpu]);
		// std::cout << "[BornShotGpu]: max model after = " << modelSliceVector[iGpu]-> max() << std::endl;
		// std::cout << "[BornShotGpu]: min model after = " << modelSliceVector[iGpu]-> min() << std::endl;
		//
		// for (int iGpu=0; iGpu<_nGpu; iGpu++){
		// 	std::cout << "[BornShotGpu]: source wavefield address at step #5 of the adjoint for gpu[" <<iGpu << "] = " << _srcWavefieldVector[iGpu] << std::endl;
		// }
	}

	// Stack models computed by each GPU
    for (int iGpu=0; iGpu<_nGpu; iGpu++){
        #pragma omp parallel for
		for (int iExt2=0; iExt2<model->getHyper()->getAxis(5).n; iExt2++){
			for (int iExt1=0; iExt1<model->getHyper()->getAxis(4).n; iExt1++){
        		for (int iy=0; iy<model->getHyper()->getAxis(3).n; iy++){
            		for (int ix=0; ix<model->getHyper()->getAxis(2).n; ix++){
                		for (int iz=0; iz<model->getHyper()->getAxis(1).n; iz++){
                    		(*model->_mat)[iExt2][iExt1][iy][ix][iz] += (*modelSliceVector[iGpu]->_mat)[iExt2][iExt1][iy][ix][iz];
                		}
            		}
        		}
			}
		}
	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateBornExtShotsGpu_3D(iGpu, _gpuList[iGpu]);
	}

}
