/** 
*  @file   larpandoracontent/LArWorkshop/TrainingExportAlgorithm.h 
* 
*  @brief  Header file for the TrainingExport algorithm class. 
* 
*  $Log: $ 
*/
#ifndef LAR_TrainingExport_ALGORITHM_H
#define LAR_TrainingExport_ALGORITHM_H 1
#include "Pandora/Algorithm.h"
#include "larpandoracontent/LArMonitoring/EventValidationAlgorithm.h"

//#ifdef MONITORING
//#include "PandoraMonitoringApi.h"
//#endif

#define IMSIZE 384 //256*1.5 Size of the generated image arrays
#define SEG 128

namespace lar_content
{
/** 
*  @brief  TrainingExportAlgorithm class 
*/
class TrainingExportAlgorithm : public EventValidationAlgorithm
{

public:
/**     
*  @brief  Factory class for instantiating algorithm     
*/class Factory : public pandora::AlgorithmFactory    
{
public:        
pandora::Algorithm *CreateAlgorithm() const;    
};

private:
pandora::StatusCode Run();
pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);
pandora::StatusCode PopulateImage(const pandora::CaloHitVector &caloHitVector, const float minX, const float minZ);
pandora::StatusCode PopulateRecoImage(const pandora::PfoList &pfoList, const pandora::CaloHitList *pCaloHitList, const pandora::CartesianVector v, const pandora::HitType TPC_VIEW);
pandora::StatusCode WriteDetectorGaps(const float minZ_U, const float minZ_V, const float minZ_W);
void fillMinimizationArray(std::array<float, 128> &hitDensity, const pandora::PfoList *const pPfoList, const pandora::CaloHitList *const pCaloHitList, const pandora::CartesianVector v, const float startD1, const float startD2, const bool directionX, const pandora::HitType TPC_VIEW);
float findMin(const std::array<float, 128> hitDensity, const float startPoint) const;
// Member variables here
pandora::StringVector m_pfoListNames;
pandora::StringVector m_clusterListNames; 
};
//------------------------------------------------------------------------------------------------------------------------------------------
inline pandora::Algorithm *TrainingExportAlgorithm::Factory::CreateAlgorithm() const
{
return new TrainingExportAlgorithm();
}
} // namespace lar_content
#endif // #ifndef LAR_TrainingExport_ALGORITHM_H
