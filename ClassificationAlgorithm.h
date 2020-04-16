/** 
*  @file   larpandoracontent/LArWorkshop/ClassificationAlgorithm.h 
* 
*  @brief  Header file for the Classification algorithm class. 
* 
*  $Log: $ 
*/
#ifndef LAR_CLASSIFICATION_ALGORITHM_H
#define LAR_CLASSIFICATION_ALGORITHM_H 1
#include "Pandora/Algorithm.h"

//#ifdef MONITORING
//#include "PandoraMonitoringApi.h"
//#endif

namespace lar_content
{
/** 
*  @brief  ClassificationAlgorithm class 
*/
class ClassificationAlgorithm : public pandora::Algorithm
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
pandora::StatusCode HitsToStringStream(const pandora::CaloHitVector caloHitVector, std::ostringstream &tempStr);
// Member variables here
std::string m_caloHitListNameU;
std::string m_caloHitListNameV;
std::string m_caloHitListNameW;
};
//------------------------------------------------------------------------------------------------------------------------------------------
inline pandora::Algorithm *ClassificationAlgorithm::Factory::CreateAlgorithm() const
{
return new ClassificationAlgorithm();
}
} // namespace lar_content
#endif // #ifndef LAR_CLASSIFICATION_ALGORITHM_H
